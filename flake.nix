{
  description = "Flake for yarqa-attn (torch-noarch CPU compartment attention)";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
      # pytest is for testshell only. Do not add it as a runtime
      # dependency of the kernel package.
      pythonCheckInputs = pkgs: with pkgs; [ pytest ];
    };
}
