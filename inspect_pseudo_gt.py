import torch
import os

# Load the pseudo_gt data
data = torch.load('./checkpoints/pseudo_gt_first_100_steps.pt')

print('=== Pseudo GT Data Structure ===')
print(f'Keys in data: {list(data.keys())}')
print()

# Check each component
for key, value in data.items():
    if value is None:
        print(f'{key}: None')
    elif isinstance(value, list):
        if len(value) > 0:
            print(f'{key}: {len(value)} items')
            if key != 'step':
                print(f'  - First item shape: {value[0].shape}')
                print(f'  - Data type: {value[0].dtype}')
        else:
            print(f'{key}: Empty list')
    print()

# Show step numbers
if 'step' in data and len(data['step']) > 0:
    steps = data['step']
    if len(steps) > 10:
        print(f'Step numbers: {steps[:10]}... (showing first 10)')
    else:
        print(f'Step numbers: {steps}')
    print(f'Total steps saved: {len(steps)}')

# File size
file_size = os.path.getsize('./checkpoints/pseudo_gt_first_100_steps.pt')
print(f'\nFile size: {file_size / (1024**2):.2f} MB')

# Show image info if available
if 'images_original' in data and len(data['images_original']) > 0:
    print('\n=== Image Information ===')
    img = data['images_original'][0]
    print(f'Image shape: {img.shape}')
    print(f'Image range: [{img.min():.3f}, {img.max():.3f}]')