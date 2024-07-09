import torch
import glob

def read_meshes_from_dir(directory, num_vertices=5023):
    """
    Reads all OBJ files within a directory and returns a list of PyTorch tensors
    representing the vertex coordinates.

    Args:
        directory: Path to the directory containing OBJ files.
        num_vertices: Expected number of vertices in each file (optional).

    Returns:
        A list of torch tensors, each representing the vertex coordinates of a
        corresponding OBJ file in the directory.
    """

    print(f"Reading .obj files in {directory}...")
    all_vertices = []
    filenames = sorted(glob.glob(f"{directory}/*.obj"))
    for filename in filenames:
        vertices = []
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts[0] == 'v':
                    # Vertex definition (v x y z)
                    vertices.append([float(p) for p in parts[1:]])

        # Check if the number of vertices matches expectations (optional)
        if len(vertices) != num_vertices:
            print(f"Warning: Expected {num_vertices} vertices in {filename}, found {len(vertices)}.")

        # Convert list to a torch tensor with desired shape
        # vertices_tensor = torch.tensor(vertices, dtype=torch.float32).unsqueeze(0)

        all_vertices.append(vertices)
        all_vertices_tensors = torch.tensor(all_vertices, dtype=torch.float32)

    print(f"Reading .obj files done! Total meshes: {len(all_vertices)}")
    return all_vertices_tensors

def main():
    directory = "/mnt/HDD3/nguyen/09_diffusion_based/diffusion-rig/mesh_input/obama_self"  # Replace with your directory path
    all_tensors = read_meshes_from_dir(directory)
    print(f"All tensor shape: {all_tensors.shape}")
    
    
if __name__ == "__main__":
    main()