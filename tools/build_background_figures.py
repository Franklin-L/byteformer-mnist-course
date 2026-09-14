from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT=Path('research/figures'); OUT.mkdir(parents=True,exist_ok=True)
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
plt.rcParams['font.family']='Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus']=False

# JPEG structure + representative byte row.
fig, ax=plt.subplots(figsize=(12,4.3),dpi=180)
ax.set_xlim(0,12); ax.set_ylim(0,4.3); ax.axis('off')
items=[('FF D8','SOI\n文件开始','#D9EAF4'),('FF DB','DQT\n量化表','#F4E5D5'),('FF C0','SOF\n尺寸/采样','#D9EAF4'),('FF C4','DHT\nHuffman表','#F4E5D5'),('FF DA','SOS\n扫描开始','#D9EAF4'),('…','Scan\n压缩数据','#E8E2F3'),('FF D9','EOI\n文件结束','#D9EAF4')]
widths=[1.25,1.55,1.55,1.55,1.55,2.15,1.45]
x=0.25
for (hexv,label,color),w in zip(items,widths):
    ax.add_patch(FancyBboxPatch((x,2.05),w,1.15,boxstyle='round,pad=.03,rounding_size=.08',facecolor=color,edgecolor='#6B8190',linewidth=1.2))
    ax.text(x+w/2,2.67,hexv,ha='center',va='center',fontsize=14,weight='bold',color='#243547')
    ax.text(x+w/2,2.25,label,ha='center',va='center',fontsize=10,color='#52636F')
    x += w+0.12
ax.annotate('',xy=(10.85,3.45),xytext=(0.45,3.45),arrowprops=dict(arrowstyle='->',lw=1.5,color='#176BA0'))
ax.text(5.65,3.68,'JPEG 文件从标记段到扫描数据按顺序组织',ha='center',fontsize=14,color='#176BA0',weight='bold')
hexrow='FF D8  FF DB  00 43  00 06  04  04  05  04  04  FF C0  00 11  08  00 1C  00 1C  FF DA  …'
ax.text(0.35,1.35,'示例字节（十六进制）',fontsize=12,color='#52636F',weight='bold')
ax.text(0.35,0.76,hexrow,fontsize=13,family='DejaVu Sans Mono',color='#243547')
ax.text(0.35,0.2,'标记字节帮助解码器定位结构；扫描数据保存真正的压缩图像内容。',fontsize=12,color='#647483')
fig.tight_layout(); fig.savefig(OUT/'jpeg_bitstream_structure.png',transparent=False,bbox_inches='tight'); plt.close(fig)

# H264 NAL sequence.
fig, ax=plt.subplots(figsize=(12,4.0),dpi=180)
ax.set_xlim(0,12); ax.set_ylim(0,4); ax.axis('off')
items=[('00 00 00 01','Start code','定位NAL','#D9EAF4',1.8),('67','SPS','序列参数','#F4E5D5',1.15),('68','PPS','图像参数','#F4E5D5',1.15),('65','IDR','关键帧','#E8E2F3',1.4),('41','P','预测帧','#DDEEDF',1.25),('01','B','双向预测','#DDEEDF',1.25),('41','P','预测帧','#DDEEDF',1.25)]
x=0.15
for value,tag,desc,color,w in items:
    ax.add_patch(FancyBboxPatch((x,1.45),w,1.1,boxstyle='round,pad=.03,rounding_size=.08',facecolor=color,edgecolor='#6B8190',linewidth=1.1))
    ax.text(x+w/2,2.15,value,ha='center',fontsize=12,weight='bold')
    ax.text(x+w/2,1.82,tag,ha='center',fontsize=12,color='#176BA0',weight='bold')
    ax.text(x+w/2,1.57,desc,ha='center',fontsize=9,color='#52636F')
    x+=w+0.12
ax.annotate('',xy=(10.95,3.15),xytext=(0.3,3.15),arrowprops=dict(arrowstyle='->',lw=1.5,color='#176BA0'))
ax.text(5.65,3.43,'H.264 将视频切成多个 NAL 单元，参数和图像帧分开保存',ha='center',fontsize=14,color='#176BA0',weight='bold')
ax.text(0.25,0.75,'SPS/PPS 受损：后续帧可能无法正确解析；IDR 受损：预测链条可能受到影响。',fontsize=12,color='#647483')
fig.tight_layout(); fig.savefig(OUT/'h264_nal_sequence.png',bbox_inches='tight'); plt.close(fig)

# Gilbert-Elliott states + burst timeline.
fig, axes=plt.subplots(1,2,figsize=(12,4.2),dpi=180,gridspec_kw={'width_ratios':[1,1.45]})
ax=axes[0]; ax.set_xlim(0,4); ax.set_ylim(0,3); ax.axis('off')
for x,y,label,color in [(1,1.65,'G\n良好状态','#DDEEDF'),(3,1.65,'B\n不良状态','#F5DEDE')]:
    ax.add_patch(FancyBboxPatch((x-.55,y-.4),1.1,.8,boxstyle='round,pad=.03,rounding_size=.08',facecolor=color,edgecolor='#6B8190',linewidth=1.2))
    ax.text(x,y,label,ha='center',va='center',fontsize=15,weight='bold',color='#243547')
ax.add_patch(FancyArrowPatch((1.55,1.85),(2.45,1.85),arrowstyle='->',mutation_scale=14,color='#BA3F3F',lw=2))
ax.add_patch(FancyArrowPatch((2.45,1.45),(1.55,1.45),arrowstyle='->',mutation_scale=14,color='#31734F',lw=2))
ax.text(2,2.15,'p：进入不良',ha='center',fontsize=11,color='#BA3F3F')
ax.text(2,1.15,'r：恢复良好',ha='center',fontsize=11,color='#31734F')
ax.text(2,.35,'r 较小时，B 状态会持续更久，形成突发错误。',ha='center',fontsize=11,color='#647483')
ax=axes[1];
steps=70; rng=np.random.default_rng(3); state=0; states=[]
for _ in range(steps):
    states.append(state)
    state = (1 if rng.random()<(.055 if state==0 else .18) else 0)
ax.step(np.arange(steps),states,where='post',color='#BA3F3F',lw=2)
ax.fill_between(np.arange(steps),0,states,step='post',alpha=.22,color='#BA3F3F')
ax.set_yticks([0,1],['G 良好','B 不良']); ax.set_xlabel('码流字节位置'); ax.set_title('一段突发损坏示意',color='#176BA0',weight='bold')
ax.grid(axis='x',alpha=.15); ax.set_ylim(-.15,1.15)
fig.suptitle('Gilbert–Elliott 二状态模型：用状态转移描述突发损坏',fontsize=15,color='#176BA0',weight='bold')
fig.tight_layout(rect=[0,0,1,.92]); fig.savefig(OUT/'gilbert_elliott_burst.png',bbox_inches='tight'); plt.close(fig)
print('generated',*[str(p) for p in sorted(OUT.glob('*structure.png'))],OUT/'gilbert_elliott_burst.png')
