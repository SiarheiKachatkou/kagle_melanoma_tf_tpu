

def build_model_str(model_code):
    effnets=['B'+str(i) for i in range(8)]
    if model_code in effnets:
        return f"efficientnet.tfkeras.EfficientNet{model_code}(weights='imagenet', include_top=False)"

    keras_apps=['ResNet50','ResNet101','Xception','MobileNetV2','MobileNet']
    if model_code in keras_apps:
        return f'tf.keras.applications.{model_code}(weights="imagenet", include_top=False)'

    raise NotImplementedError(f' model_code {model_code} must be in list {effnets+keras_apps}')