import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gripper


class GripperTests(unittest.TestCase):
    def test_move_goal_and_failure_handling(self):
        for opening, success in ((True, True), (False, True), (False, False)):
            with self.subTest(opening=opening, success=success):
                node = Mock()
                ros = Mock()
                ros.create_node.return_value = node
                client = Mock()
                handle = Mock(accepted=True)
                submission = Mock()
                submission.done.return_value = True
                submission.result.return_value = handle
                client.send_goal_async.return_value = submission
                result = Mock()
                result.done.return_value = True
                result.result.return_value = SimpleNamespace(status=4, result=SimpleNamespace(success=success, error="blocked"))
                handle.get_result_async.return_value = result
                fake_modules = {"rclpy": ros, "rclpy.action": SimpleNamespace(ActionClient=Mock(return_value=client)),
                                "franka_msgs.action": SimpleNamespace(Move=SimpleNamespace(Goal=SimpleNamespace)),
                                "action_msgs.msg": SimpleNamespace(GoalStatus=SimpleNamespace(STATUS_SUCCEEDED=4))}
                args = SimpleNamespace(open=opening, action="/panda_gripper/move", speed=0.05, timeout=15)
                with patch.dict(sys.modules, fake_modules):
                    if success:
                        gripper.operate(args)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "blocked"):
                            gripper.operate(args)
                goal = client.send_goal_async.call_args.args[0]
                self.assertEqual(goal.width, 0.08 if opening else 0)
                self.assertEqual(goal.speed, 0.05)
                node.destroy_node.assert_called_once()
                ros.shutdown.assert_called_once()

    def test_invalid_arguments_send_no_commands(self):
        for argv in (["--open", "--close"], ["--close", "--speed", "nan"], []):
            with self.subTest(argv=argv), patch.object(sys, "argv", ["gripper.py", *argv]), \
                 patch.object(gripper.subprocess, "run") as run:
                with self.assertRaises(SystemExit):
                    gripper.main()
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
