import torch
import torch_scatter
import torch_geometric
import pandas
import numpy
import esm

def main():
    print("Hello from immunofoundation!")
    print(torch.__version__)
    print(torch_scatter.__version__)
    print(torch_geometric.__version__)
    print(torch.cuda.is_available())
    print(numpy.__version__)
    print(pandas.__version__)
    print(esm.__version__)




if __name__ == "__main__":
    main()
