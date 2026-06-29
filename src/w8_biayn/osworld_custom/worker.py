from __future__ import annotations

import json
import sys
from pathlib import Path

from .registry import load_task
from .harness import _cleanup_record, _container_ids, config_from_dict, create_desktop_env, import_desktop_env, run_episode_in_process, write_json


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print('usage: python -m w8_biayn.osworld_custom.worker <input.json> <output.json>', file=sys.stderr)
        return 2

    input_path = Path(args[0])
    output_path = Path(args[1])
    payload = json.loads(input_path.read_text(encoding='utf-8'))
    config = config_from_dict(payload['config'])
    run_dir = Path(payload['run_dir'])
    actions = list(payload.get('actions') or ['WAIT'])

    before_containers = _container_ids()
    DesktopEnv = import_desktop_env(config.osworld_path or Path.cwd())
    env = create_desktop_env(DesktopEnv, config)
    rows: list[dict] = []
    try:
        for item in payload['tasks']:
            task = load_task(item['task_config_path'], domain=item['domain'])
            artifact_dir = run_dir / 'domains' / task.domain / task.task_id
            rows.append(run_episode_in_process(env, task, actions, artifact_dir, config))
    finally:
        close = getattr(env, 'close', None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        write_json(run_dir / 'container_cleanup.json', _cleanup_record(config, before_containers))

    output_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
