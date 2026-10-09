"""Tests for application logic that does not need a physical camera."""

import unittest

from tennis_vision.app import FpsMeter, parse_args


class FpsMeterTests(unittest.TestCase):
    def test_reports_measured_rate(self) -> None:
        meter = FpsMeter(smoothing=1.0)
        self.assertEqual(meter.tick(now=1.0), 0.0)
        self.assertAlmostEqual(meter.tick(now=1.05), 20.0)

    def test_rejects_invalid_smoothing(self) -> None:
        with self.assertRaises(ValueError):
            FpsMeter(smoothing=0.0)


class ArgumentTests(unittest.TestCase):
    def test_camera_and_device_are_configurable(self) -> None:
        args = parse_args(["--camera", "2", "--device", "mps"])
        self.assertEqual(args.camera, 2)
        self.assertEqual(args.device, "mps")

    def test_video_and_independent_inference_cadences_are_configurable(self) -> None:
        args = parse_args(
            ["--video", "clip.mp4", "--pose-every", "2", "--ball-every", "3"]
        )
        self.assertEqual(args.video, "clip.mp4")
        self.assertEqual(args.pose_every, 2)
        self.assertEqual(args.ball_every, 3)

    def test_tracknet_backend_and_checkpoint_are_selectable(self) -> None:
        args = parse_args(
            [
                "--ball-detector",
                "tracknet",
                "--tracknet-checkpoint",
                "custom.pth",
            ]
        )
        self.assertEqual(args.ball_detector, "tracknet")
        self.assertEqual(args.tracknet_checkpoint, "custom.pth")

    def test_rfdetr_backend_checkpoint_and_resolution_are_selectable(self) -> None:
        args = parse_args(
            [
                "--ball-detector",
                "rfdetr",
                "--rfdetr-checkpoint",
                "fine-tuned.pth",
                "--rfdetr-input-size",
                "1024",
            ]
        )
        self.assertEqual(args.ball_detector, "rfdetr")
        self.assertEqual(args.rfdetr_checkpoint, "fine-tuned.pth")
        self.assertEqual(args.rfdetr_input_size, 1024)

    def test_video_export_and_headless_preview_are_configurable(self) -> None:
        args = parse_args(
            [
                "--video",
                "clip.mov",
                "--output-video",
                "outputs/annotated.mp4",
                "--output-fps",
                "30",
                "--no-preview",
            ]
        )
        self.assertEqual(args.output_video, "outputs/annotated.mp4")
        self.assertEqual(args.output_fps, 30.0)
        self.assertTrue(args.no_preview)


if __name__ == "__main__":
    unittest.main()
