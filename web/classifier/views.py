import sys
from pathlib import Path

from django.shortcuts import render

from classifier.forms import UploadImageForm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.predictor import predict_image


def home(request):
    """Render the landing page with an upload form."""
    form = UploadImageForm()
    return render(request, "classifier/home.html", {"form": form})


def predict_view(request):
    """Handle image upload and show the prediction result."""
    if request.method != "POST":
        return render(request, "classifier/home.html", {"form": UploadImageForm()})

    form = UploadImageForm(request.POST, request.FILES)
    if not form.is_valid():
        return render(request, "classifier/home.html", {"form": form})

    uploaded_image = request.FILES["image"]
    temp_path = Path("/tmp") / uploaded_image.name
    with temp_path.open("wb+") as file_handle:
        for chunk in uploaded_image.chunks():
            file_handle.write(chunk)

    try:
        prediction, confidence = predict_image(temp_path)
    except FileNotFoundError as exc:
        return render(
            request,
            "classifier/result.html",
            {"error": str(exc)},
        )
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return render(
        request,
        "classifier/result.html",
        {"prediction": prediction, "confidence": confidence},
    )
