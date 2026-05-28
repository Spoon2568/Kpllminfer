import torch
import kp_allocator
rank = 0
hbm_numa = (rank % 16) + 16
kp_allocator.init_allocator(hbm_numa, False)
kp_allocator.enable()

kp_allocator.set_allocate_on_hbm(True)
try:
    # bf16 [4096, 4096] ≈ 32MB，满足 >512KB
    weight = torch.empty(4096, 4096, dtype=torch.bfloat16)
    weight.uniform_(-0.1, 0.1)
finally:
    kp_allocator.set_allocate_on_hbm(False)

ptr = weight.untyped_storage().data_ptr()
on_hbm = kp_allocator.get_alloc_id_on_hbm(ptr) != -1
print(kp_allocator.is_on_hbm(weight.data_ptr()))
