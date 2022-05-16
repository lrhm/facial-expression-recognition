

from .webcam_classifier import video_capture



def custom_test(model):
    """
    Custom test function for the model.
    """
    print("Custom test function for the model.")

    video_capture(model)

    return