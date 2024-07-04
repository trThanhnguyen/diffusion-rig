import pickle
    
def read_flame(path):
    print("Reading custom flame codes...")
    with open(path, 'rb') as f:
        codes = pickle.load(f)
    # print(type(codes)) # dict
    # print(type(codes["expression"][0])) # <class 'numpy.ndarray'>
    # print(codes["expression"][0]) # <class 'numpy.ndarray'>
    return codes

def main():
    path = "/mnt/HDD3/nguyen/09_diffusion_based/diffusion-rig/infer_flame_codes/flame_M003_Neutral_0.pkl"
    flame_codes = read_flame(path)

if __name__ == "__main__":
    main()