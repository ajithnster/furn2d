"""Runs one floor-plan image conversion for the showroom API (separate process, own venv with
numpy/scipy/pillow). Usage: plan_import_worker.py <image> <workdir>. Prints JSON to stdout:
{"project": {...}, "labels": [...], "scaleFrom": "..."}; the API deletes <workdir> afterwards."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'plan_import'))
from plan_to_openplan import ask_gemini_rest, convert  # noqa: E402

image, work = sys.argv[1], Path(sys.argv[2])
report = convert(image, work / 'plan.json', work, ask=ask_gemini_rest)
print(json.dumps({
    'project': json.loads((work / 'plan.json').read_text()),
    'labels': json.loads((work / 'labels.json').read_text()),
    'scaleFrom': report.get('scale_from', ''),
}))
