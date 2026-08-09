# Third-party notices

This repository contains an adapted LEAP Hand MuJoCo model under
`models/leap_hand/`. Its original license is preserved at
`models/leap_hand/LICENSE`.

The implementation and mapping approach were informed by:

- [Vision-based Teleoperation with Dexterous Hand in MuJoCo](https://github.com/fzhang327/Vision-based-Teleoperation-with-Dexterous-Hand-in-MuJoCo)
- [LeapHand Teleoperation Repo](https://github.com/Julianxng/LeapHand-Teleoperation-Repo)
- [LEAP Hand API](https://github.com/leap-hand/LEAP_Hand_API)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- [ROBOTIS Dynamixel SDK](https://github.com/ROBOTIS-GIT/DynamixelSDK)
- [RealSense SDK 2.0](https://github.com/realsenseai/librealsense)

Those projects retain their own copyright and license terms.

The optional D455 source follows the Apache-2.0 RealSense Python wrapper
conventions for simultaneous color/depth streams and
`rs.align(rs.stream.color)`. It uses the separately installed `pyrealsense2`
package; no RealSense SDK source is copied into this repository.
