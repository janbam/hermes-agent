"""Tests for the /pause CLI command."""

from types import SimpleNamespace
from unittest.mock import Mock, patch


def _import_cli():
    import hermes_cli.config as config_mod

    if not hasattr(config_mod, "save_env_value_secure"):
        config_mod.save_env_value_secure = lambda key, value: {
            "success": True,
            "stored_as": key,
            "validated": False,
        }

    import cli as cli_mod
    return cli_mod


def _make_cli(*, running=True, agent=None, accepting_pause=True):
    return SimpleNamespace(
        _agent_running=running,
        _agent_accepting_pause=accepting_pause,
        agent=agent,
    )


def test_pause_command_requests_running_agent_pause():
    cli_mod = _import_cli()
    agent = SimpleNamespace(pause_after_tool=Mock(return_value=True))
    stub = _make_cli(running=True, agent=agent)

    with patch.object(cli_mod, "_cprint") as mock_cprint:
        assert cli_mod.HermesCLI.process_command(stub, "/pause") is True

    agent.pause_after_tool.assert_called_once_with()
    printed = " ".join(str(c) for c in mock_cprint.call_args_list)
    assert "Pause requested" in printed


def test_pause_command_reports_when_already_requested():
    cli_mod = _import_cli()
    agent = SimpleNamespace(pause_after_tool=Mock(return_value=False))
    stub = _make_cli(running=True, agent=agent)

    with patch.object(cli_mod, "_cprint") as mock_cprint:
        assert cli_mod.HermesCLI.process_command(stub, "/pause") is True

    printed = " ".join(str(c) for c in mock_cprint.call_args_list)
    assert "already requested" in printed


def test_pause_command_noops_when_idle():
    cli_mod = _import_cli()
    agent = SimpleNamespace(pause_after_tool=Mock(return_value=True))
    stub = _make_cli(running=False, agent=agent)

    with patch.object(cli_mod, "_cprint") as mock_cprint:
        assert cli_mod.HermesCLI.process_command(stub, "/pause") is True

    agent.pause_after_tool.assert_not_called()
    printed = " ".join(str(c) for c in mock_cprint.call_args_list)
    assert "nothing to pause" in printed


def test_pause_command_noops_after_agent_model_loop_finished():
    cli_mod = _import_cli()
    agent = SimpleNamespace(pause_after_tool=Mock(return_value=True), resume_after_pause=Mock(return_value=True))
    stub = SimpleNamespace(_agent_running=True, _agent_accepting_pause=False, agent=agent)

    with patch.object(cli_mod, "_cprint") as mock_cprint:
        assert cli_mod.HermesCLI.process_command(stub, "/pause") is True

    agent.pause_after_tool.assert_not_called()
    agent.resume_after_pause.assert_called_once_with()
    printed = " ".join(str(c) for c in mock_cprint.call_args_list)
    assert "finished its model/tool loop" in printed


def test_pause_command_reports_ctrl_backtick_for_already_requested():
    cli_mod = _import_cli()
    agent = SimpleNamespace(pause_after_tool=Mock(return_value=False))
    stub = _make_cli(running=True, agent=agent)

    with patch.object(cli_mod, "_cprint") as mock_cprint:
        assert cli_mod.HermesCLI.process_command(stub, "/pause") is True

    printed = " ".join(str(c) for c in mock_cprint.call_args_list)
    assert "already requested" in printed
    assert "Ctrl+`" in printed


def test_running_inline_dispatch_accepts_pause_and_steer_only_when_busy():
    cli_mod = _import_cli()
    busy = SimpleNamespace(_agent_running=True)
    idle = SimpleNamespace(_agent_running=False)

    assert cli_mod.HermesCLI._should_handle_running_command_inline(busy, "/pause") is True
    assert cli_mod.HermesCLI._should_handle_running_command_inline(busy, "/steer wait") is True
    assert cli_mod.HermesCLI._should_handle_running_command_inline(busy, "/queue later") is False
    assert cli_mod.HermesCLI._should_handle_running_command_inline(idle, "/pause") is False


def test_resume_agent_pause_lifts_pending_pause():
    cli_mod = _import_cli()
    agent = SimpleNamespace(resume_after_pause=Mock(return_value=True))
    stub = _make_cli(running=True, agent=agent)

    with patch.object(cli_mod, "_cprint") as mock_cprint:
        assert cli_mod.HermesCLI._resume_agent_pause(stub) is True

    agent.resume_after_pause.assert_called_once_with()
    printed = " ".join(str(c) for c in mock_cprint.call_args_list)
    assert "Pause lifted" in printed


def test_dashboard_config_schema_excludes_pause_busy_mode():
    from hermes_cli.web_server import CONFIG_SCHEMA

    assert "pause" not in CONFIG_SCHEMA["display.busy_input_mode"]["options"]
