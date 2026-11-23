def calculate_parameters(layers: list[dict]) -> int:
    total = 0
    for layer in layers:

        if layer['type'] == 'dense':
            w = layer['input_size'] * layer['output_size']
            if layer.get('bias', True):        # missing key → defaults to True
                w += layer['output_size']
            total += w

        elif layer['type'] == 'conv2d':
            w = layer['in_channels'] * layer['out_channels'] * layer['kernel_size']**2
            if layer.get('bias', True):
                w += layer['out_channels']
            total += w

    return total