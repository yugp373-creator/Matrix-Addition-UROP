using CUDA, Random

# Just to verify that julia is running on GPU and not CPU. 

device = CUDA.device()
println("Using GPU device: ", device)


# Switched to floating point numbers becuase GPU's run faster on them. 
# Pre-allocate to save memory
# Switched to 10000x10000 matrices to take advantage of full GPU power

a = CuArray{Float32}(undef, 10000, 10000)
b = CuArray{Float32}(undef, 10000, 10000)
c = CuArray{Float32}(undef, 10000, 10000)

function loop_matrix_addition(n)
    for x in 1:n
        CUDA.rand!(a)    # generates floating point numbers from 0-1
        CUDA.rand!(b)
        c .= (a .+ b) .* 1000 # adds a and b and multiplies the sum by 1000 using element-wise multiplication
    end
end

#Timer


start_time = time()
loop_matrix_addition(10000)
CUDA.synchronize()  # I put this to ensure time is only after all GPU operations are finished)
end_time = time()

println("Completed 10,000 additions in $(round(end_time - start_time; digits=2)) seconds")

#86.96 seconds