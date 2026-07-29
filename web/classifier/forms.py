from django import forms


class UploadImageForm(forms.Form):
    """Simple form for uploading a leaf image."""

    image = forms.ImageField(
        label="Upload a leaf image",
        help_text="Please upload a JPG or PNG image.",
    )
