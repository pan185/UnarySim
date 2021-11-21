# %%
import torch
from UnarySim.kernel.linear import *
import matplotlib.pyplot as plt
import time
import math
import numpy as np

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
def test(rounding = "round", abs_err = True, input_width = 4, weight_width = 8):
    ufc_err_min_list = []
    ufc_err_max_list = []
    ufc_err_mean_list = []
    ufc_err_std_list = []
    
    ifc_err_min_list = []
    ifc_err_max_list = []
    ifc_err_mean_list = []
    ifc_err_std_list = []

    cfc_err_min_list = []
    cfc_err_max_list = []
    cfc_err_mean_list = []
    cfc_err_std_list = []
    
    x_label = []
    
    for et_cycle in range(1, 2**input_width):
        bitwidth_tuple = (input_width+1, weight_width+1)

        # equivalent bitwidth tuple for hubflex and fxp
        input_width_eq = None
        if et_cycle == 1: input_width_eq = 2
        elif et_cycle == 3: input_width_eq = 3
        elif et_cycle == 7: input_width_eq = 4
        elif et_cycle == 15: input_width_eq = 5
        bitwidth_tuple_eq = (input_width_eq, weight_width+1)

        print(f"*** et_cycle={et_cycle}\nbitwidth_tuple for tlut=({input_width+1},{weight_width+1})")
        if input_width_eq != None:
            print(f"bitwidth_tuple for baseline=({input_width_eq},{weight_width+1})")
        
        in_feature = 2
        out_feature = 2**12
        bias = False
        
        input = torch.cat(2*[(torch.arange(0, out_feature)/out_feature - 0.5).unsqueeze(1)], 1).to(device)
        input[:, 1] = 0.

        fc = torch.nn.Linear(in_feature, out_feature, bias=bias).to(device)
        fc.weight.data = torch.cat(2*[(torch.arange(0, out_feature)/out_feature - 0.5).unsqueeze(1)], 1).to(device)
        fc.weight.data[:, 1] = 0.
        fc_o = fc(input)

        # ufc = HUBLinear(in_feature, out_feature, bias=bias, binary_weight=fc.weight.data, binary_bias=fc.bias, cycle=cycle, rounding=rounding).to(device)
        if input_width_eq != None:
            ufc = HUBLinear_flex(in_feature, out_feature, bias=bias, binary_weight=fc.weight.data, binary_bias=fc.bias, bitwidth=bitwidth_tuple_eq, rounding=rounding).to(device)
            ufc_o = ufc(input)

        cfc = TlutLinear(in_feature, out_feature, bias=bias, binary_weight=fc.weight.data, binary_bias=fc.bias, cycle=et_cycle, bitwidth=bitwidth_tuple, rounding=rounding).to(device)
        cfc_o = cfc(input)
        
        if input_width_eq != None:
            ifc = FxpLinear(in_feature, out_feature, bias=bias, binary_weight=fc.weight.data, binary_bias=fc.bias, bitwidth=bitwidth_tuple_eq, keep_res="input",  more_res="input", rounding=rounding).to(device)
            ifc_o = ifc(input)
        
        if abs_err is True:
            if input_width_eq != None: ufc_err = (ufc_o - fc_o)
            cfc_err = (cfc_o - fc_o)
            if input_width_eq != None: ifc_err = (ifc_o - fc_o)
        else:
            if input_width_eq != None: ufc_err = (ufc_o - fc_o) / fc_o
            cfc_err = (cfc_o - fc_o) / fc_o
            if input_width_eq != None: ifc_err = (ifc_o - fc_o) / fc_o
        
        if input_width_eq != None:
            ufc_err_min_list.append(np.nanmin(ufc_err.cpu().detach().numpy()))
            ufc_err_max_list.append(np.nanmax(ufc_err.cpu().detach().numpy()))
            ufc_err_mean_list.append(np.nanmean(np.abs(ufc_err.cpu().detach().numpy())))
            ufc_err_std_list.append(np.nanstd(ufc_err.cpu().detach().numpy()))
        else:
            ufc_err_min_list.append(np.nan)
            ufc_err_max_list.append(np.nan)
            ufc_err_mean_list.append(np.nan)
            ufc_err_std_list.append(np.nan)

        cfc_err_min_list.append(np.nanmin(cfc_err.cpu().detach().numpy()))
        cfc_err_max_list.append(np.nanmax(cfc_err.cpu().detach().numpy()))
        cfc_err_mean_list.append(np.nanmean(np.abs(cfc_err.cpu().detach().numpy())))
        cfc_err_std_list.append(np.nanstd(cfc_err.cpu().detach().numpy()))

        if input_width_eq != None:
            ifc_err_min_list.append(np.nanmin(ifc_err.cpu().detach().numpy()))
            ifc_err_max_list.append(np.nanmax(ifc_err.cpu().detach().numpy()))
            ifc_err_mean_list.append(np.nanmean(np.abs(ifc_err.cpu().detach().numpy())))
            ifc_err_std_list.append(np.nanstd(ifc_err.cpu().detach().numpy()))
        else:
            ifc_err_min_list.append(np.nan)
            ifc_err_max_list.append(np.nan)
            ifc_err_mean_list.append(np.nan)
            ifc_err_std_list.append(np.nan)

        x_label.append(et_cycle)
    return ufc_err_min_list, ufc_err_max_list, ufc_err_mean_list, ufc_err_std_list, cfc_err_min_list, cfc_err_max_list, cfc_err_mean_list, cfc_err_std_list, ifc_err_min_list, ifc_err_max_list, ifc_err_mean_list, ifc_err_std_list, x_label


# %%
rounding = "round"
abs_err = True
input_wd = 4
wght_wd = 8
ufc_err_min_list, ufc_err_max_list, ufc_err_mean_list, ufc_err_std_list, cfc_err_min_list, cfc_err_max_list, cfc_err_mean_list, cfc_err_std_list, ifc_err_min_list, ifc_err_max_list, ifc_err_mean_list, ifc_err_std_list, x_label = test(rounding, abs_err, input_wd, wght_wd)
print(ufc_err_mean_list)
print(ufc_err_std_list)
print()

print(cfc_err_mean_list)
print(cfc_err_std_list)
print()

print(ifc_err_mean_list)
print(ifc_err_std_list)
print()

print(x_label)

# %%
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

font = {'family':'Times New Roman', 'size': 6}

matplotlib.rc('font', **font)

my_dpi = 300
fig_h = 1
fig_w = 3.3115

# construct some data like what you have:
x = np.array([i for i in range(len(ufc_err_mean_list))])
means1 = np.array(ufc_err_mean_list)
stds1 = np.array(ufc_err_std_list)
mins1 = np.array(ufc_err_min_list)
maxs1 = np.array(ufc_err_max_list)

means_ = np.array(cfc_err_mean_list)
stds_ = np.array(cfc_err_std_list)
mins_ = np.array(cfc_err_min_list)
maxs_ = np.array(cfc_err_max_list)

means3 = np.array(ifc_err_mean_list)
stds3 = np.array(ifc_err_std_list)
mins3 = np.array(ifc_err_min_list)
maxs3 = np.array(ifc_err_max_list)

# x_label = ['5-16', '6-32', '7-64', '8-128', '9-256']
x_label = range(1, 2**input_wd)

width = 0.20
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

ax.plot(x, means1, "-o", label="uSystolic", color="#7A81FF", ms=4)
ax.fill_between(x, means1-stds1, means1+stds1, alpha=0.3, color="#7A81FF", edgecolor=None)

ax.plot(x, means_, "-o", label="Temporal-LUT", color="#6ACCBC", ms=4)
ax.fill_between(x, means_-stds_, means_+stds_, alpha=0.3, color="#6ACCBC", edgecolor=None)

# ax.plot(x, means2, "-s", label="FXP-o-res", color="#FF7F7F", ms=4)
# ax.fill_between(x, means2-stds2, means2+stds2, alpha=0.3, color="#FF7F7F", edgecolor=None)

ax.plot(x, means3, "-^", label="FXP", color="#D783FF", ms=4)
ax.fill_between(x, means3-stds3, means3+stds3, alpha=0.3, color="#D783FF", edgecolor=None)

ax.set_xticks(x)
ax.set_xticklabels(x_label)
ax.set_yscale('linear')
ax.set_yticks([0, 0.02, 0.04, 0.06])
ax.set_yticklabels(["0.00", "0.02", "0.04", "0.06"])
ax.set_ylim(0, 0.07)
ax.legend(loc="upper right", ncol=3, frameon=False)

fig.tight_layout()
plt.show()
path = "/home/zhewen/Repo/UnarySim/test/kernel/"
fig.savefig(path + f"test_kernel_linear_fxp_hub_tlut_compare_abs_err_input{input_wd}_wght{wght_wd}.pdf", bbox_inches='tight', dpi=my_dpi, pad_inches=0.02)


# %%
rounding = "round"
abs_err = False
ufc_err_min_list, ufc_err_max_list, ufc_err_mean_list, ufc_err_std_list, cfc_err_min_list, cfc_err_max_list, cfc_err_mean_list, cfc_err_std_list, ifc_err_min_list, ifc_err_max_list, ifc_err_mean_list, ifc_err_std_list, x_label = test(rounding, abs_err, input_wd, wght_wd)
print(ufc_err_mean_list)
print(ufc_err_std_list)
print()

print(cfc_err_mean_list)
print(cfc_err_std_list)
print()

print(ifc_err_mean_list)
print(ifc_err_std_list)
print()

print(x_label)

# %%
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

font = {'family':'Times New Roman', 'size': 6}

matplotlib.rc('font', **font)

my_dpi = 300
fig_h = 1
fig_w = 3.3115

# construct some data like what you have:
x = np.array([i for i in range(len(ufc_err_mean_list))])
means1 = np.array(ufc_err_mean_list)
stds1 = np.array(ufc_err_std_list)
mins1 = np.array(ufc_err_min_list)
maxs1 = np.array(ufc_err_max_list)

means_ = np.array(cfc_err_mean_list)
stds_ = np.array(cfc_err_std_list)
mins_ = np.array(cfc_err_min_list)
maxs_ = np.array(cfc_err_max_list)

means3 = np.array(ifc_err_mean_list)
stds3 = np.array(ifc_err_std_list)
mins3 = np.array(ifc_err_min_list)
maxs3 = np.array(ifc_err_max_list)

# x_label = ['5-16', '6-32', '7-64', '8-128', '9-256']

width = 0.20
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

ax.plot(x, means1, "-o", label="uSystolic", color="#7A81FF", ms=4)
ax.fill_between(x, means1-stds1, means1+stds1, alpha=0.3, color="#7A81FF", edgecolor=None)

ax.plot(x, means_, "-o", label="Temporal-LUT", color="#6ACCBC", ms=4)
ax.fill_between(x, means_-stds_, means_+stds_, alpha=0.3, color="#6ACCBC", edgecolor=None)

# ax.plot(x, means2, "-s", label="FXP-o-res", color="#FF7F7F", ms=4)
# ax.fill_between(x, means2-stds2, means2+stds2, alpha=0.3, color="#FF7F7F", edgecolor=None)

ax.plot(x, means3, "-^", label="FXP", color="#D783FF", ms=4)
ax.fill_between(x, means3-stds3, means3+stds3, alpha=0.3, color="#D783FF", edgecolor=None)

ax.set_xticks(x)
ax.set_xticklabels(x_label)
ax.set_yscale('linear')
ax.set_yticks([0, 0.4, 0.8])
ax.set_yticklabels(["0.00", "0.40", "0.80"])
# ax.legend(loc="upper right", ncol=3, frameon=False)

fig.tight_layout()
plt.show()
path = "/home/zhewen/Repo/UnarySim/test/kernel/"
fig.savefig(path + f"test_kernel_linear_fxp_hub_tlut_compare_rel_err_et_input{input_wd}_wght{wght_wd}.pdf", bbox_inches='tight', dpi=my_dpi, pad_inches=0.02)


# %%
