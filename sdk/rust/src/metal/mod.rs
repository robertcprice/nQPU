// Metal Integration Module
// =========================
// Handles Metal device, command queues, and shader compilation

use cocoa::base::id;
use cocoa::foundation::NSString as cocoaNSString;
use metal::*;
use objc::{msg_send, sel, sel_impl, class, runtime::Object};
use std::ffi::CStr;
use std::path::Path;
use std::ptr;

pub struct MetalDevice {
    pub device: id,
    pub command_queue: id,
    pub library: id,
}

impl MetalDevice {
    /// Create a new Metal device with command queue
    pub fn new() -> Result<Self, String> {
        unsafe {
            // Get the default Metal device
            let device: id = msg_send![class!(MTLCreateSystemDefaultDevice), newDevice];
            if device.is_null() {
                return Err("Failed to create Metal device".to_string());
            }

            // Create command queue
            let command_queue: id = msg_send![device, newCommandQueue];
            if command_queue.is_null() {
                return Err("Failed to create command queue".to_string());
            }

            // Load shader library from bundled shaders
            let library = Self::load_library(device)?;

            Ok(MetalDevice {
                device,
                command_queue,
                library,
            })
        }
    }

    /// Load Metal shader library
    fn load_library(device: id) -> Result<id, String> {
        unsafe {
            // Try to load from framework bundle first
            let framework_path = std::env::var("FRAMEWORK_PATH")
                .unwrap_or_else(|_| "/System/Library/Frameworks".to_string());

            // For now, we'll compile from source string
            // In production, you'd load from a .metallib file
            let library_path = std::env::current_exe()
                .map_err(|e| format!("Failed to get exe path: {}", e))?
                .parent()
                .ok_or("No parent directory")?
                .join("shaders.metal");

            // If shaders.metal exists next to binary, load it
            let library = if library_path.exists() {
                Self::compile_from_file(device, &library_path)?
            } else {
                // Otherwise, load from embedded source (simplified for now)
                return Err(format!("Shader file not found: {:?}", library_path));
            };

            Ok(library)
        }
    }

    fn compile_from_file(device: id, path: &Path) -> Result<id, String> {
        unsafe {
            let shader_source = std::fs::read_to_string(path)
                .map_err(|e| format!("Failed to read shader file: {}", e))?;

            let ns_string = cocoaNSString::alloc(nil).init_str(&shader_source);

            let compile_options: id = msg_send![class!(MTLCompileOptions), alloc];
            let compile_options: id = msg_send![compile_options, init];

            let library: id = msg_send![device,
                newLibraryWithSource:ns_string
                options:compile_options
                error:ptr::null_mut::<Object>()
            ];

            if library.is_null() {
                return Err("Failed to compile Metal shaders".to_string());
            }

            Ok(library)
        }
    }

    /// Create a compute command encoder
    pub fn create_command_buffer(&self) -> id {
        unsafe {
            msg_send![self.command_queue, commandBuffer]
        }
    }

    /// Get a compute pipeline state for a function
    pub fn get_pipeline_state(&self, function_name: &str) -> Result<id, String> {
        unsafe {
            let ns_name = cocoaNSString::alloc(nil).init_str(function_name);

            let function: id = msg_send![self.library, newFunctionWithName:ns_name];
            if function.is_null() {
                return Err(format!("Function not found: {}", function_name));
            }

            let pipeline: id = msg_send![self.device,
                newComputePipelineStateWithFunction:function
                error:ptr::null_mut::<Object>()
            ];

            if pipeline.is_null() {
                return Err(format!("Failed to create pipeline for: {}", function_name));
            }

            Ok(pipeline)
        }
    }
}

impl Drop for MetalDevice {
    fn drop(&mut self) {
        // Metal uses ARC, so we don't need to manually release
    }
}

pub struct MetalBuffer {
    pub buffer: id,
    pub length: usize,
}

impl MetalBuffer {
    /// Create a new Metal buffer
    pub fn new<T>(device: id, data: &[T], options: MTLResourceOptions) -> Result<Self, String>
    where
        T: Copy,
    {
        unsafe {
            let byte_length = std::mem::size_of_val(data);
            let buffer: id = msg_send![device,
                newBufferWithBytes:data.as_ptr()
                length:byte_length
                options:options
            ];

            if buffer.is_null() {
                return Err("Failed to create Metal buffer".to_string());
            }

            Ok(MetalBuffer {
                buffer,
                length: data.len(),
            })
        }
    }

    /// Create an empty buffer
    pub fn uninit(device: id, length: usize, options: MTLResourceOptions) -> Result<Self, String> {
        unsafe {
            let byte_length = length * std::mem::size_of::<f32>();
            let buffer: id = msg_send![device,
                newBufferWithLength:byte_length
                options:options
            ];

            if buffer.is_null() {
                return Err("Failed to create Metal buffer".to_string());
            }

            Ok(MetalBuffer { buffer, length })
        }
    }

    /// Copy buffer contents to host
    pub fn copy_to_host<T>(&self, data: &mut [T]) -> Result<(), String>
    where
        T: Copy,
    {
        if data.len() != self.length {
            return Err("Buffer size mismatch".to_string());
        }

        unsafe {
            let contents = msg_send![self.buffer, contents] as *const T;
            std::ptr::copy_nonoverlapping(contents, data.as_mut_ptr(), self.length);
        }

        Ok(())
    }
}

#[link(name = "Metal", kind = "framework")]
#[link(name = "Foundation", kind = "framework")]
extern "C" {}

// Metal constants
pub const MTLResourceOptions: usize = 0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_device_creation() {
        let device = MetalDevice::new();
        assert!(device.is_ok(), "Metal device creation failed");
    }
}
