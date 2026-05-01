"""Tests for soft pause-after-tool control in AIAgent."""

import threading

from run_agent import AIAgent


def _bare_agent() -> AIAgent:
    agent = object.__new__(AIAgent)
    agent._pause_after_tool_requested = False
    agent._pause_after_tool_lock = threading.Lock()
    agent._pause_resume_event = threading.Event()
    agent._pause_resume_event.set()
    agent._interrupt_requested = False
    agent._pending_steer = None
    agent._pending_steer_lock = threading.Lock()
    agent.quiet_mode = True
    agent._emit_status = lambda message: None
    agent._safe_print = lambda *args, **kwargs: None
    return agent


class TestPauseAfterToolFlag:
    def test_pause_after_tool_sets_flag_once(self):
        agent = _bare_agent()

        assert agent.pause_after_tool() is True
        assert agent._has_pause_after_tool_request() is True

        assert agent.pause_after_tool() is False
        assert agent._has_pause_after_tool_request() is True

    def test_consume_pause_after_tool_clears_flag(self):
        agent = _bare_agent()
        agent.pause_after_tool()

        assert agent._consume_pause_after_tool_request() is True
        assert agent._has_pause_after_tool_request() is False
        assert agent._consume_pause_after_tool_request() is False

    def test_pause_after_tool_works_for_object_new_stub_without_lock(self):
        agent = object.__new__(AIAgent)

        assert agent.pause_after_tool() is True
        assert agent._has_pause_after_tool_request() is True
        assert agent.pause_after_tool() is False
        assert agent._consume_pause_after_tool_request() is True
        assert agent._has_pause_after_tool_request() is False


class TestPauseAfterToolInterruptInteraction:
    def test_clear_interrupt_clears_pending_pause_request(self):
        agent = _bare_agent()
        agent._interrupt_requested = True
        agent._interrupt_message = "stop"
        agent._interrupt_thread_signal_pending = False
        agent._execution_thread_id = None
        agent._tool_worker_threads = set()
        agent._tool_worker_threads_lock = threading.Lock()
        agent._pending_steer = None
        agent._pending_steer_lock = threading.Lock()

        agent.pause_after_tool()
        assert agent._has_pause_after_tool_request() is True

        agent.clear_interrupt()

        assert agent._has_pause_after_tool_request() is False


class TestPauseResumeGate:
    def test_resume_after_pause_clears_pending_pause_request(self):
        agent = _bare_agent()
        agent.pause_after_tool()

        assert agent._has_pause_after_tool_request() is True
        assert agent.resume_after_pause() is True
        assert agent._has_pause_after_tool_request() is False
        assert agent.resume_after_pause() is False

    def test_wait_for_pause_resume_blocks_until_resumed(self):
        agent = _bare_agent()
        agent.pause_after_tool()
        finished = threading.Event()

        def _wait():
            assert agent._wait_for_pause_resume_if_requested() is True
            finished.set()

        thread = threading.Thread(target=_wait)
        thread.start()
        assert finished.wait(0.05) is False

        assert agent.resume_after_pause() is True
        assert finished.wait(1.0) is True
        thread.join(timeout=1.0)
        assert agent._has_pause_after_tool_request() is False

    def test_wait_for_pause_resume_returns_immediately_without_request(self):
        agent = _bare_agent()

        assert agent._wait_for_pause_resume_if_requested() is False

    def test_pause_notice_emitted_once(self):
        agent = _bare_agent()
        agent.quiet_mode = False
        statuses = []
        prints = []
        agent._emit_status = statuses.append
        agent._safe_print = lambda *args, **kwargs: prints.append(args)
        agent.pause_after_tool()

        finished = threading.Event()

        def _wait():
            assert agent._wait_for_pause_resume_if_requested() is True
            finished.set()

        thread = threading.Thread(target=_wait)
        thread.start()
        assert statuses and "Paused before next model call" in statuses[0]
        assert prints == []
        agent.resume_after_pause()
        assert finished.wait(1.0) is True
        thread.join(timeout=1.0)

    def test_late_steer_typed_while_paused_appends_to_latest_tool_result(self):
        agent = _bare_agent()
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "call_1", "function": {"name": "terminal"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "output: hermes"},
        ]
        agent.pause_after_tool()

        finished = threading.Event()

        def _wait():
            assert agent._wait_for_pause_resume_if_requested() is True
            finished.set()

        thread = threading.Thread(target=_wait)
        thread.start()
        assert finished.wait(0.05) is False

        assert agent.steer("next run pwd") is True
        assert agent.resume_after_pause() is True
        assert finished.wait(1.0) is True
        thread.join(timeout=1.0)

        agent._apply_pending_steer_to_tool_results(messages, len(messages))

        assert "User guidance: next run pwd" in messages[-1]["content"]
        assert agent._drain_pending_steer() is None
