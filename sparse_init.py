import torch, math


def _layer_infos(modules):
    import torch.nn as nn

    infos = []
    for module in modules:
        if isinstance(module, nn.Linear):
            fan_out, fan_in = module.weight.shape
            infos.append((module, fan_in, fan_out, module.weight.numel()))
        elif isinstance(module, nn.Conv2d):
            out_channels, in_channels, kernel_h, kernel_w = module.weight.shape
            infos.append((module, in_channels, out_channels + kernel_h + kernel_w, module.weight.numel()))
    return infos


def _apply_row_sparsity(tensor, layer_sparsity, init_type='uniform', generator=None, min_active: int = 0):
    if not (0.0 <= layer_sparsity < 1.0):
        raise ValueError("layer_sparsity must be in [0,1)")

    with torch.no_grad():
        if tensor.ndimension() == 2:
            rows, row_width = tensor, tensor.shape[1]
        elif tensor.ndimension() == 4:
            out_channels, in_channels, height, width = tensor.shape
            rows, row_width = tensor.view(out_channels, -1), in_channels * height * width
        else:
            raise ValueError("Only 2D (Linear) and 4D (Conv2d) tensors are supported")

        scale = math.sqrt(1.0 / row_width)
        if init_type == 'uniform':
            rows.uniform_(-scale, scale)
        elif init_type == 'normal':
            rows.normal_(0, scale)
        else:
            raise ValueError("Unknown initialization type")

        num_zeros = int(math.ceil(layer_sparsity * row_width))
        if min_active > 0:
            num_zeros = min(num_zeros, max(0, row_width - min_active))
        for row_idx in range(rows.shape[0]):
            rows[row_idx, torch.randperm(row_width, generator=generator)[:num_zeros]] = 0


def sparse_init(modules, target_sparsity, type='uniform', seed=None,
             min_density: float = 0.0, max_density: float = 1.0,
             min_active: int = 0):
    if not (0.0 <= target_sparsity < 1.0):
        raise ValueError("target_sparsity must be in [0,1)")

    layer_infos = _layer_infos(modules)
    if not layer_infos:
        return {}

    target_nonzeros = (1 - target_sparsity) * sum(param_count for _, _, _, param_count in layer_infos)
    dense_layers = set()
    remaining = set(layer_infos)
    epsilon = 0.0

    while remaining:
        nonzeros_dense = sum(param_count for _, _, _, param_count in dense_layers)
        sum_fans = sum(fan_in + fan_out for _, fan_in, fan_out, _ in remaining)
        epsilon = (target_nonzeros - nonzeros_dense) / max(sum_fans, 1e-12)
        new_dense = [
            info for info in remaining
            if epsilon * (info[1] + info[2]) / info[3] >= 1.0
        ]
        if not new_dense:
            break
        dense_layers.update(new_dense)
        remaining.difference_update(new_dense)

    generator = torch.Generator() if seed is None else torch.Generator().manual_seed(seed)
    layer_sparsities = {}
    for module, fan_in, fan_out, param_count in layer_infos:
        density = 1.0 if (module, fan_in, fan_out, param_count) in dense_layers else epsilon * (fan_in + fan_out) / param_count
        density = max(min_density, min(density, max_density, 1.0))
        sparsity = 1 - density
        layer_sparsities[module] = sparsity
        _apply_row_sparsity(module.weight, sparsity, init_type=type, generator=generator, min_active=min_active)
        if getattr(module, 'bias', None) is not None:
            module.bias.data.zero_()

    return layer_sparsities
