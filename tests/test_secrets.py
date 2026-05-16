from __future__ import annotations

import json

from w8_biayn.secrets import default_bucket_for_project, get_project_id


def test_get_project_id_from_service_account(tmp_path):
    path = tmp_path / "sa.json"
    path.write_text(
        json.dumps(
            {
                "type": "service_account",
                "project_id": "my-project",
                "private_key": "SECRET",
            }
        ),
        encoding="utf-8",
    )

    assert get_project_id(path) == "my-project"


def test_default_bucket_for_project_normalizes_underscore():
    assert default_bucket_for_project("My_Project") == "gs://my-project-w8-biayn"

