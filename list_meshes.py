import airsim
import sys


def list_scene_objects(client, name_regex=None):
    try:
        if name_regex:
            return sorted(client.simListSceneObjects(name_regex) or [])
        return sorted(client.simListSceneObjects() or [])
    except TypeError:
        return sorted(client.simListSceneObjects() or [])


def get_mesh_info(client, name):
    info = {}

    if hasattr(client, "simGetMeshPositionVertexBuffer"):
        try:
            mesh = client.simGetMeshPositionVertexBuffer(name)
            if hasattr(mesh, "vertices"):
                info["vertex_count"] = len(mesh.vertices)
            elif hasattr(mesh, "positions"):
                info["vertex_count"] = len(mesh.positions)
            else:
                info["vertex_count"] = len(mesh)
        except Exception as exc:
            info["mesh_error"] = str(exc)

    try:
        pose = client.simGetObjectPose(name)
        info["pose"] = pose
    except Exception as exc:
        info["pose_error"] = str(exc)

    return info


def format_pose(pose):
    pos = pose.position
    ori = pose.orientation
    return (
        f"pos=({pos.x_val:.3f}, {pos.y_val:.3f}, {pos.z_val:.3f}) "
        f"ori=({ori.w_val:.3f}, {ori.x_val:.3f}, {ori.y_val:.3f}, {ori.z_val:.3f})"
    )


def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()

    name_regex = sys.argv[1] if len(sys.argv) > 1 else None
    objects = list_scene_objects(client, name_regex)

    print(f"Found {len(objects)} scene objects.")
    for name in objects:
        print(f"- {name}")
        info = get_mesh_info(client, name)
        if "vertex_count" in info:
            print(f"  vertices: {info['vertex_count']}")
        if "pose" in info:
            print(f"  {format_pose(info['pose'])}")
        if "mesh_error" in info:
            print(f"  mesh_error: {info['mesh_error']}")
        if "pose_error" in info:
            print(f"  pose_error: {info['pose_error']}")


if __name__ == "__main__":
    main()
