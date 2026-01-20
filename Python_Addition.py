import torch
import time


# Just to verify  that python is running on GPU and not CPU. 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# I pre-allocated tensors to save memory
# Increased matrix size to 10000x10000 to take advantage of full GPU power.  

a = torch.empty((10000, 10000), dtype=torch.float32, device=device)
b = torch.empty_like(a)
c = torch.empty_like(a)

torch.cuda.synchronize()
start_time = time.time()


for _ in range(10000):
    a.random_(0, 1000)   #fills in a/b which are empty with randomly generated numbers
    b.random_(0, 1000) 
    c.copy_(a)           #replaced c with a and then adds b, I am not completely sure why this is optimal but it was one of the improvements chatgpt recommended and it worked quite well. 
    c.add_(b)            

torch.cuda.synchronize()
end_time = time.time()

print(f"Completed 10,000 additions in {round(end_time - start_time, 2)} seconds")

#115.14 seconds