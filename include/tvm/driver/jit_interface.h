#define EXPORT_DLL __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
    EXPORT_DLL tvm::runtime::Module TVMCompile(const std::string& onnx_txt, const std::string& target, const std::string& target_host, int opt_level, const std::vector<std::vector<int64_t>>& input_shapes);
    EXPORT_DLL void TVMExtractOutputShapes(tvm::runtime::Module& mod, size_t num_outputs, std::vector<std::vector<int64_t>>& output_shapes);

    EXPORT_DLL void TVMRun(tvm::runtime::Module& mod, std::vector<DLTensor>& inputs, std::vector<DLTensor>& outputs);
    
    
}  // TVM_EXTERN_C
#endif
