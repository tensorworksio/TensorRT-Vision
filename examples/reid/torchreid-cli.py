import torch
import argparse
import torchreid


def parse_arguments():
    parser = argparse.ArgumentParser(description="TorchReID Model CLI")
    parser.add_argument("-l", "--list", action="store_true", help="List all available models")
    parser.add_argument("-m", "--model", type=str, help="Specify model name for details or export")
    parser.add_argument("-e", "--export", action="store_true", help="Export the specified model to ONNX")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output path for ONNX model")
    parser.add_argument("-s", "--size", type=int, nargs=2, default=[256, 128], help="Input size (width height)")
    return parser


def main(parser):
    args = parser.parse_args()
    if args.list:
        torchreid.models.show_avai_models()
    elif args.model:
        model = torchreid.models.build_model(name=args.model, 
                                             num_classes=1000, 
                                             loss='softmax', 
                                             pretrained=True)
        size = (1, 3, args.size[0], args.size[1])
        torchreid.utils.compute_model_complexity(model, size, verbose=True)
        if args.export:
            output = args.output if args.output else args.model + ".onnx"
            export(model, output, size)
    else:
        parser.print_help()


def export(model, onnx_path, size=(1, 3, 256, 128)):
    model.eval()
    # Dummy input
    dummy = torch.randn(size)
    # Export
    torch.onnx.export(model,
                      dummy,
                      onnx_path,
                      export_params=True,
                      opset_version=12,
                      do_constant_folding=True,
                      input_names=['image'],
                      output_names=['feature'],
                      dynamic_axes={'image': {0: 'batch_size'},
                                    'feature': {0: 'batch_size'}},
                      )


if __name__ == "__main__":
    parser = parse_arguments()
    main(parser)