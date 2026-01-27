import time

import viser
import h5py
import viser

############################
# Load trajectory
############################
def load_h5(path):
    with h5py.File(path, "r") as f:
        data = {k: f[k][...] for k in f.keys()}
    return data


############################
# Main
############################
def main(
    h5_path: str,
    object_urdf: str,
    robot_urdf: str,
):
    traj = load_h5(h5_path)

    T = traj["robot_pos"].shape[0]

    server = viser.ViserServer()
    print("Viser running at:", server.url)

    ################################
    # Load URDFs
    ################################
    object_vis = server.add_urdf(
        name="object",
        urdf_path=object_urdf,
        position=traj["object_pos"][0],
        orientation=traj["object_quat"][0],
    )

    robot_vis = server.add_urdf(
        name="leap_hand",
        urdf_path=robot_urdf,
        position=traj["robot_pos"][0],
        orientation=traj["robot_quat"][0],
    )

    ################################
    # UI controls
    ################################
    with server.gui_folder("Playback"):
        play = server.gui_checkbox("Play", initial_value=False)
        reverse = server.gui_checkbox("Reverse", initial_value=False)
        t_slider = server.gui_slider(
            "Frame",
            min=0,
            max=T - 1,
            step=1,
            initial_value=0,
        )
        fps = server.gui_slider(
            "FPS",
            min=1,
            max=120,
            step=1,
            initial_value=30,
        )

    ################################
    # Playback loop
    ################################
    t = 0
    last_time = time.time()

    while True:
        time.sleep(0.001)

        if play.value:
            now = time.time()
            dt = now - last_time
            if dt >= 1.0 / fps.value:
                step = -1 if reverse.value else 1
                t = (t + step) % T
                t_slider.value = t
                last_time = now
        else:
            t = int(t_slider.value)

        ################################
        # Update object
        ################################
        object_vis.set_pose(
            position=traj["object_pos"][t],
            orientation=traj["object_quat"][t],
        )

        object_vis.set_joint_positions(
            traj["object_joints"][t]
        )

        ################################
        # Update robot
        ################################
        robot_vis.set_pose(
            position=traj["robot_pos"][t],
            orientation=traj["robot_quat"][t],
        )

        robot_vis.set_joint_positions(
            traj["robot_joints"][t]
        )


if __name__ == "__main__":
    main(
        h5_path="/home/nl455/spider/example_datasets/processed/arctic/inspire_hand/bimanual/box-30-230/0/trajectory_dexmachina.h5",
        object_urdf="object.urdf",
        robot_urdf="leap_hand.urdf",
    )
