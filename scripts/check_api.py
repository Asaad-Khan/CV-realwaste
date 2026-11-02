"""Check the running API using real HTTP requests."""

import argparse
import json
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(
            "data/raw/realwaste-main/RealWaste/Cardboard/Cardboard_100.jpg"
        ),
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    results = []

    def check(name, response, expected_status):
        passed = response.status_code == expected_status
        results.append({
            "check": name,
            "expected_status": expected_status,
            "actual_status": response.status_code,
            "passed": passed,
        })
        print(
            f"{'PASS' if passed else 'FAIL'}: {name} "
            f"(HTTP {response.status_code})"
        )
        if not passed:
            print(response.text[:500])
        return passed

    def upload(session, filename, payload, content_type):
        return session.post(
            f"{base_url}/predict",
            files={"file": (filename, payload, content_type)},
            timeout=(5, 60),
        )

    with requests.Session() as session:
        response = session.get(
            f"{base_url}/health", timeout=(5, 30)
        )
        if check("Model readiness", response, 200):
            if response.json().get("status") != "ready":
                raise RuntimeError("Health endpoint did not report readiness.")

        with args.image.open("rb") as image_file:
            response = upload(
                session, args.image.name, image_file, "image/jpeg"
            )
        if check("Valid image", response, 200):
            prediction = response.json()
            items = prediction["top_predictions"]

            if (
                len(items) != 3
                or prediction["predicted_class"] != items[0]["label"]
                or not all(0 <= item["score"] <= 1 for item in items)
                or any(
                    items[i]["score"] < items[i + 1]["score"]
                    for i in range(len(items) - 1)
                )
            ):
                raise RuntimeError("Prediction response failed schema checks.")

        response = upload(session, "empty.jpg", b"", "image/jpeg")
        check("Empty file", response, 400)

        # An image filename and MIME type must not make text a valid image.
        response = upload(
            session, "fake.jpg", b"This is not an image.", "image/jpeg"
        )
        check("Invalid image content", response, 400)

        buffer = BytesIO()
        Image.new("RGB", (32, 32), color="white").save(buffer, format="BMP")
        response = upload(
            session, "unsupported.bmp", buffer.getvalue(), "image/bmp"
        )
        check("Unsupported image format", response, 415)

        response = upload(
            session,
            "oversized.jpg",
            b"x" * (5 * 1024 * 1024 + 1),
            "image/jpeg",
        )
        check("File above 5 MiB", response, 413)

        # This compresses to a small file but exceeds the decoded pixel limit.
        buffer = BytesIO()
        with Image.new("RGB", (3500, 3500), color="white") as image:
            image.save(buffer, format="PNG")
        response = upload(
            session, "large_dimensions.png", buffer.getvalue(), "image/png"
        )
        check("Image above 12 million pixels", response, 413)

        response = session.post(
            f"{base_url}/predict", timeout=(5, 30)
        )
        check("Missing upload field", response, 422)

        response = session.get(
            f"{base_url}/health", timeout=(5, 30)
        )
        check("Service responds after invalid requests", response, 200)

    report_path = Path("reports/api_checks.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {"base_url": base_url, "checks": results},
            indent=2,
        ),
        encoding="utf-8",
    )

    failures = sum(not result["passed"] for result in results)
    print(f"\nReport saved to: {report_path}")

    if failures:
        raise SystemExit(f"{failures} API checks failed.")

    print(f"All {len(results)} API checks passed.")


if __name__ == "__main__":
    main()