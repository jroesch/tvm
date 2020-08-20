#define EXPORT_DLL __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
    EXPORT_DLL tvm::runtime::Module TVMCompile(const std::string& onnx_txt, const std::string& target, const std::string& target_host, int opt_level);
    EXPORT_DLL void TVMRun(tvm::runtime::Module& mod,  std::vector<DLTensor> inputs, std::vector<DLTensor> outputs, tvm::runtime::TVMRetValue* ret);
}  // TVM_EXTERN_C
#endif
