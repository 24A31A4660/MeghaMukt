from preprocessing.transforms import get_train_transforms


def test_train_transforms_include_brightness_noise_and_crop():
    transform = get_train_transforms(256)
    transform_names = [type(t).__name__ for t in transform.transforms]

    assert "HorizontalFlip" in transform_names
    assert "VerticalFlip" in transform_names
    assert "RandomBrightnessContrast" in transform_names
    assert "GaussianNoise" in transform_names
    assert "RandomResizedCrop" in transform_names
