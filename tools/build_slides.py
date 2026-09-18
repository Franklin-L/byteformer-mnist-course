"""Build the editable Chinese course deck for student-guided experiments."""
import argparse
import gzip
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'docs/assets'
RESEARCH_FIGURES = ROOT / 'research/figures'
BLUE, INK, LIGHT, RED, GRAY = '176BA0', '243547', 'EDF4F8', 'BA3F3F', '647483'
FONT = 'Noto Sans CJK SC'
REPO = 'https://github.com/Franklin-L/byteformer-mnist-course'
KAGGLE = 'https://www.kaggle.com/code'
AUTODL = 'https://www.autodl.com/'


def rgb(code): return RGBColor.from_string(code)


def text(slide, x, y, w, h, value, size=22, color=INK, bold=False, align=None, font=FONT):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(.03)
    tf.margin_top = tf.margin_bottom = Inches(.02)
    for i, line in enumerate(str(value).split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.name = font; p.font.size = Pt(size); p.font.bold = bold; p.font.color.rgb = rgb(color)
        p.space_after = Pt(6)
        if align is not None: p.alignment = align
    return shape


def rect(slide, x, y, w, h, fill, line=None, rounded=False):
    shape=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid(); shape.fill.fore_color.rgb=rgb(fill)
    shape.line.fill.background() if line is None else None
    if line is not None: shape.line.color.rgb=rgb(line)
    return shape


def link(slide, x, y, w, label, url, size=18):
    shape=text(slide,x,y,w,.42,label,size,BLUE)
    for run in shape.text_frame.paragraphs[0].runs:
        run.hyperlink.address=url; run.font.underline=True
    return shape


def picture(slide, path, x, y, w, h):
    with Image.open(path) as im: iw,ih=im.size
    ratio=min(w/iw,h/ih); nw,nh=iw*ratio,ih*ratio
    return slide.shapes.add_picture(str(path), Inches(x+(w-nw)/2), Inches(y+(h-nh)/2), width=Inches(nw), height=Inches(nh))


class Deck:
    def __init__(self):
        self.prs=Presentation(); self.prs.slide_width=Inches(13.333); self.prs.slide_height=Inches(7.5)
        self.prs.core_properties.title='字节域语义内容理解实验说明'
        self.prs.core_properties.author='Franklin-L'
        self.prs.core_properties.subject='字节模型与ByteFormer码流图像分类实验'
    def slide(self,title,subtitle=''):
        s=self.prs.slides.add_slide(self.prs.slide_layouts[6])
        picture(s,ASSETS/'hust_wordmark.png',.55,.2,2.65,.52)
        text(s,8.0,.24,4.7,.35,'字节域语义内容理解',13,GRAY,align=PP_ALIGN.RIGHT)
        text(s,.6,.86,12.1,.56,title,30,INK,True)
        rect(s,.65,1.5,12.02,.023,'BACBD7')
        if subtitle: text(s,.66,1.66,12,.55,subtitle,17,GRAY)
        text(s,12.1,7.0,.55,.3,str(len(self.prs.slides)),11,GRAY,align=PP_ALIGN.RIGHT)
        return s
    def bullets(self,title,items,subtitle=''):
        s=self.slide(title,subtitle)
        start=2.35 if subtitle else 2.0
        height=min(1.35,4.5/len(items))
        for i,(heading,body) in enumerate(items):
            y=start+i*height
            rect(s,.7,y+.08,.09,.25,BLUE)
            text(s,.94,y,11.7,.38,heading,22,INK,True)
            text(s,.96,y+.4,11.55,height-.36,body,18,GRAY)
        return s
    def command(self,title,where,command,success,tip=''):
        s=self.slide(title,where)
        lines=command.split('\n'); boxh=max(1.0,min(2.5,.33*len(lines)+.45))
        rect(s,.75,2.36,11.8,boxh,'20384B',rounded=True)
        code_shape=text(s,1.0,2.54,11.25,boxh-.32,command,18,'FFFFFF',font='DejaVu Sans Mono')
        for paragraph in code_shape.text_frame.paragraphs: paragraph.space_after=Pt(2)
        text(s,.8,2.66+boxh,11.5,.5,'成功标志',23,BLUE,True)
        text(s,.83,3.18+boxh,11.4,.8,success,21)
        if tip: text(s,.83,4.15+boxh,11.4,.55,tip,17,RED)
        return s
    def table(self,title,headers,rows,widths,subtitle=''):
        s=self.slide(title,subtitle)
        y=2.35 if subtitle else 2.05
        h=min(4.5,.6*(len(rows)+1))
        table=s.shapes.add_table(len(rows)+1,len(headers),Inches(.75),Inches(y),Inches(11.8),Inches(h)).table
        for j,w in enumerate(widths): table.columns[j].width=Inches(w)
        for i,row in enumerate([headers]+rows):
            for j,value in enumerate(row):
                c=table.cell(i,j);c.text=str(value);c.margin_left=Inches(.13);c.margin_top=Inches(.1)
                c.fill.solid();c.fill.fore_color.rgb=rgb(BLUE if i==0 else ('EDF4F8' if i%2 else 'FFFFFF'))
                for p in c.text_frame.paragraphs:
                    p.font.name=FONT;p.font.size=Pt(17);p.font.bold=i==0;p.font.color.rgb=rgb('FFFFFF' if i==0 else INK)
        return s


def digits_asset():
    p=ASSETS/'mnist_digits.png'
    raw=ROOT/'data/MNIST/raw'
    images=gzip.open(raw/'t10k-images-idx3-ubyte.gz','rb').read()[16:]
    labels=gzip.open(raw/'t10k-labels-idx1-ubyte.gz','rb').read()[8:]
    canvas=Image.new('RGB',(1100,440),'white'); draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',25)
    for digit in range(10):
        i=labels.index(digit);im=Image.frombytes('L',(28,28),images[i*784:(i+1)*784]).resize((160,160),Image.Resampling.NEAREST)
        x=(digit%5)*220+30;y=(digit//5)*220+8
        canvas.paste(im,(x,y));draw.text((x+72,y+170),str(digit),font=font,fill='#176ba0')
    canvas.save(p);return p


def codebox(slide,x,y,w,h,command,size=17):
    rect(slide,x,y,w,h,'20384B',rounded=True)
    shape=text(slide,x+.18,y+.14,w-.36,h-.28,command,size,'FFFFFF',font='DejaVu Sans Mono')
    for p in shape.text_frame.paragraphs:p.space_after=Pt(3)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft',action='store_true')
    parser.add_argument('--pdf',action='store_true')
    args=parser.parse_args()
    d=Deck()
    # 1. Cover, preserving the teacher's visual template.
    s=d.prs.slides.add_slide(d.prs.slide_layouts[6])
    picture(s,ASSETS/'campus.png',0,0,13.333,2.5);rect(s,0,2.53,13.333,.055,BLUE)
    picture(s,ASSETS/'hust_seal.png',.7,3.2,1.8,1.8)
    text(s,2.8,3.23,10.0,1.13,'字节域语义内容理解实验说明',34,BLUE,True)
    text(s,2.83,4.68,4.65,.55,'助教：李方成',24,'000000',True)
    text(s,8.0,4.68,4.65,.55,'教师：吴科君',24,'000000',True)
    text(s,2.83,5.46,4.65,.55,'lifangcheng2002@163.com',19,'000000',True)
    text(s,8.0,5.46,4.65,.55,'kjwu@hust.edu.cn',19,'000000',True)
    # 2. Multimedia communication and the role of a bitstream.
    s=d.slide('01 多媒体通信与码流','图像和视频经过压缩编码后，以二进制码流的形式传输、存储和解析。')
    flow=[('图像 / 视频\n信源',.72),('压缩编码\nJPEG / H.264',3.15),('信道或\n存储介质',5.58),('解码与\n视觉分析',8.01),('分类 / 检测\n内容理解',10.44)]
    for i,(label,x) in enumerate(flow):
        rect(s,x,2.12,2.05,.86,LIGHT,rounded=True)
        text(s,x+.07,2.30,1.91,.5,label,18,BLUE,True,PP_ALIGN.CENTER)
        if i<4:text(s,x+2.08,2.35,.31,.35,'→',20,BLUE,True,PP_ALIGN.CENTER)
    text(s,.82,3.42,5.35,.42,'像素、压缩文件和字节序列',22,INK,True)
    text(s,.86,4.02,5.45,1.46,'同一幅图像可以有不同的表示：\n· 像素矩阵：适合传统视觉网络\n· JPEG 文件：包含标记、表和扫描数据\n· 字节序列：统一处理图像、视频和其他文件',18)
    text(s,6.65,3.42,5.55,.42,'码流中包含哪些信息',22,INK,True)
    rect(s,6.63,4.03,2.61,1.20,'DDEEDF',rounded=True)
    text(s,6.80,4.20,2.27,.39,'结构信息',19,'31734F',True,PP_ALIGN.CENTER)
    text(s,6.80,4.66,2.27,.48,'文件标记、参数表\n帧类型',14,GRAY,align=PP_ALIGN.CENTER)
    rect(s,9.63,4.03,2.61,1.20,'E8E2F3',rounded=True)
    text(s,9.80,4.20,2.27,.39,'内容信息',19,'5A4A86',True,PP_ALIGN.CENTER)
    text(s,9.80,4.66,2.27,.48,'扫描数据、图像与\n视频内容',14,GRAY,align=PP_ALIGN.CENTER)
    text(s,6.73,5.56,5.36,.90,'码流既是传输和存储的载体，\n也是理解多媒体内容的一种表示。',19,BLUE,True)
    # 3. JPEG structure.
    s=d.slide('02 JPEG：从文件标记到扫描数据','JPEG 不是一串无意义的数字；不同段落共同描述图像如何被解码。')
    picture(s,RESEARCH_FIGURES/'jpeg_bitstream_structure.png',.76,1.94,11.82,3.72)
    text(s,.84,5.83,11.55,.88,'SOI / EOI 标记文件边界；DQT 和 DHT 保存解码所需的表；SOS 之后是压缩扫描数据。\n标记段或扫描数据受损，可能导致解码失败、错块或花屏。',18)
    # 4. H.264 structure.
    s=d.slide('03 H.264：参数、关键帧与预测帧','视频码流由多个 NAL 单元组成，帧之间还存在时间上的依赖关系。')
    picture(s,RESEARCH_FIGURES/'h264_nal_sequence.png',.76,1.98,11.82,3.70)
    text(s,.84,5.87,11.55,.77,'SPS / PPS 决定后续帧如何解析；IDR 是可以独立解码的关键帧；P、B 帧依赖参考帧。\n一处错误可能沿着后续帧传播，这也是视频码流分析比单张图更复杂的原因。',18)
    # 5. Gilbert-Elliott burst error model.
    s=d.slide('04 突发损坏：Gilbert–Elliott 二状态模型','错误不一定独立均匀出现；网络拥塞、存储故障等因素会让错误连续发生。')
    picture(s,RESEARCH_FIGURES/'gilbert_elliott_burst.png',.70,1.92,7.20,3.95)
    text(s,8.18,2.12,4.15,.45,'模型的两个状态',22,BLUE,True)
    rect(s,8.18,2.78,3.96,.88,'DDEEDF',rounded=True)
    text(s,8.35,2.98,3.62,.46,'G：良好状态，错误率低',18,'31734F',True,PP_ALIGN.CENTER)
    rect(s,8.18,3.87,3.96,.88,'F5DEDE',rounded=True)
    text(s,8.35,4.07,3.62,.46,'B：不良状态，错误率高',18,RED,True,PP_ALIGN.CENTER)
    text(s,8.19,5.03,4.03,1.01,'p：G → B 的转移概率\nr：B → G 的恢复概率\nr 越小，突发错误持续越久。',18)
    # 5. Why understand bytes directly after the channel model.
    s=d.slide('05 码流损坏的影响：为什么需要字节域理解','损坏位置不同，解码器可能直接失败，也可能输出不可靠的像素或后续帧。')
    text(s,.82,1.96,11.3,.4,'损坏位置 → 解码结果 → 对视觉模型的影响',22,BLUE,True,PP_ALIGN.CENTER)
    cards=[('文件头 / 参数表','JPEG 的 SOI、DQT、DHT；\nH.264 的 SPS、PPS','无法解析文件或后续帧','F5DEDE'),('扫描数据 / 关键帧','JPEG 压缩数据；\nH.264 的 IDR、P、B 帧','错块、花屏、错误传播','F4E5D5'),('解码输出','像素缺失、尺寸异常或\n整段视频无法恢复','像素域模型没有可靠输入','E8E2F3')]
    for i,(heading,where,result,color) in enumerate(cards):
        x=.78+i*4.02
        rect(s,x,2.56,3.72,2.15,color,rounded=True)
        text(s,x+.16,2.78,3.40,.36,heading,20,BLUE,True,PP_ALIGN.CENTER)
        text(s,x+.18,3.31,3.36,.56,where,15,INK,align=PP_ALIGN.CENTER)
        text(s,x+.18,4.14,3.36,.39,result,15,RED,True,align=PP_ALIGN.CENTER)
    picture(s,RESEARCH_FIGURES/'byteaction_framework.png',.77,5.08,5.35,1.33)
    picture(s,RESEARCH_FIGURES/'vub_motivation.png',6.99,5.12,5.32,1.16)
    text(s,.92,6.49,11.38,.43,'传统流程：码流 → 解码 → 像素 → 视觉模型　　字节域流程：码流 → 字节模型 → 分类或语义理解',16,BLUE,True,PP_ALIGN.CENTER)
    # 6. General byte models only.
    s=d.slide('06 代表性字节模型','字节模型直接读取 0—255 字节值，重点解决超长序列和多尺度结构建模。')
    models=[
        ('MEGABYTE','2023','全局模型处理字节块\n局部模型预测块内字节','层级生成'),
        ('ByteFormer','2023','卷积缩短序列\n窗口 Transformer 分类','文件分类'),
        ('MambaByte','2024','选择性状态空间模型\n近似线性处理长序列','长序列建模'),
        ('bGPT','2024','Patch-level + Byte-level\n多模态字节生成','通用字节生成'),
        ('mBLM','2025','多层级 Byte Patch\nTransformer / Mamba 可选','百万字节上下文'),
    ]
    for i,(name,year,body,status) in enumerate(models):
        x=.65+i*2.53
        rect(s,x,2.22,2.28,3.23,LIGHT,rounded=True)
        text(s,x+.11,2.43,2.06,.42,name,20,BLUE,True,PP_ALIGN.CENTER)
        text(s,x+.11,2.88,2.06,.3,year,15,GRAY,align=PP_ALIGN.CENTER)
        text(s,x+.14,3.45,2.0,.93,body,16,INK,align=PP_ALIGN.CENTER)
        text(s,x+.14,4.73,2.0,.45,status,13,GRAY,align=PP_ALIGN.CENTER)
    rect(s,.83,5.85,11.69,.63,'DDEBF3',rounded=True)
    text(s,1.01,6.02,11.34,.34,'课程选择 ByteFormer：面向文件分类，有公开代码和预训练权重，适合完成 MNIST 微调。',20,BLUE,True,PP_ALIGN.CENTER)
    # 8. ByteFormer framework and the course classification task.
    s=d.slide('07 ByteFormer框架与课程任务')
    picture(s,RESEARCH_FIGURES/'byteformer_model_arch.png',.68,1.92,5.25,3.9)
    stages=[('MNIST','图像 / JPEG'),('字节序列','0—255'),('ByteFormer','预训练主干'),('分类结果','数字 0—9')]
    for i,(heading,body) in enumerate(stages):
        x=6.25+i*1.53
        rect(s,x,2.16,1.28,1.08,LIGHT,rounded=True)
        text(s,x+.05,2.31,1.18,.32,heading,12,BLUE,True,PP_ALIGN.CENTER)
        text(s,x+.05,2.72,1.18,.28,body,13,GRAY,align=PP_ALIGN.CENTER)
        if i<3:text(s,x+1.29,2.49,.22,.3,'→',18,BLUE,True,PP_ALIGN.CENTER)
    text(s,6.31,3.64,5.65,1.88,'· 使用公开的ByteFormer Tiny预训练权重\n· 将分类头改为10个类别\n· 在训练集更新参数，用验证集选择模型\n· 最后在独立测试集上报告结果',19)
    rect(s,.82,6.12,11.64,.58,'DDEBF3',rounded=True)
    text(s,1.02,6.27,11.24,.32,'基础任务：完成码流图像分类、参数对比和结果分析。',20,BLUE,True,PP_ALIGN.CENTER)
    # 9. Balanced 1/10 MNIST split.
    s=d.slide('08 课程数据集与三类划分','课程仓库已经提供固定划分，所有同学使用相同样本。')
    picture(s,digits_asset(),.69,2.12,5.0,3.94)
    splits=[('训练集','5,000','每类500张\n包含原图与轻微旋转、平移视图'),('验证集','1,000','每类100张\n用于选择最佳模型'),('测试集','1,000','每类100张\n训练结束后独立评估')]
    for i,(name,count,body) in enumerate(splits):
        x=6.02+i*2.13
        rect(s,x,2.18,1.89,3.18,LIGHT,rounded=True)
        text(s,x+.11,2.41,1.67,.38,name,22,BLUE,True,PP_ALIGN.CENTER)
        text(s,x+.11,3.02,1.67,.48,count,27,INK,True,PP_ALIGN.CENTER)
        text(s,x+.13,3.79,1.63,1.0,body,15,GRAY,align=PP_ALIGN.CENTER)
    text(s,6.14,5.76,6.12,.61,'数据文件：data/course_1of10/　测试集不参与训练和模型选择。',16,GRAY)
    # 10. GPU platforms and external tutorials on one page.
    s=d.table('09 实验环境与在线平台',['运行平台','适用情况','入口'],[
        ['Kaggle免费GPU','推荐；导入课程Notebook后运行','Notebook + Internet + GPU'],
        ['AutoDL租用GPU','Kaggle额度不足时使用','选择PyTorch镜像，打开Terminal'],
        ['本地GPU','已安装Python 3.9—3.12','pip install -r requirements-local.txt']], [2.45,5.1,4.25],subtitle='平台界面、免费额度和租用价格可能调整，以平台当前页面为准。')
    links=[
        ('知乎｜Kaggle GPU资源使用教程','https://zhuanlan.zhihu.com/p/18209757723'),
        ('CSDN｜Kaggle平台使用指导','https://blog.csdn.net/yyyyyybw/article/details/148336854'),
        ('Kaggle官方Notebook文档','https://www.kaggle.com/docs/notebooks'),
        ('AutoDL官方快速开始','https://www.autodl.com/docs/quick_start/'),
    ]
    for i,(label,url) in enumerate(links):
        link(s,.87+(i%2)*6.0,5.42+(i//2)*.58,5.65,label,url,16)
    # 11. Code, data, and pretrained weights.
    s=d.slide('10 获取代码、数据与预训练权重','课程Notebook中已经写好相同命令。')
    command=f'!git clone {REPO}.git\n%cd /kaggle/working/byteformer-mnist-course\n!python -m pip install -r requirements.txt\n!python prepare.py'
    codebox(s,.8,2.24,11.73,1.92,command,17)
    text(s,.87,4.49,11.6,.55,'看到[READY]即准备完成；课程数据集随仓库提供，ByteFormer预训练权重自动下载并校验。',19)
    text(s,.88,5.31,11.6,1.14,'train_course_subset.py：训练　evaluate_course_corruption.py：测试\npredict.py：单图预测　data/course_1of10/：固定数据　outputs/：实验结果',18,GRAY)
    # 12. Student-selected training settings.
    s=d.slide('11 模型训练与参数设置')
    settings=[('训练 / 验证 / 测试','5,000 / 1,000 / 1,000'),('训练轮数','根据验证结果自行设置'),('batch size','自行设置，参考32'),('学习率','主干1e-4；分类头10倍'),('输入增强','轻微旋转与水平平移')]
    for i,(a,b) in enumerate(settings):
        y=2.01+i*.69;rect(s,.8,y,5.53,.59,LIGHT);text(s,.94,y+.1,2.24,.4,a,18,BLUE,True);text(s,3.23,y+.1,2.95,.43,b,17)
    codebox(s,6.67,2.05,5.82,2.18,'EPOCHS = int(input("epochs: "))\nBATCH_SIZE = int(input("batch: "))\n!python train_course_subset.py \\\n  --method clean --epochs {EPOCHS} \\\n  --batch-size {BATCH_SIZE} --clean-augmentations \\\n  --output outputs/course_clean',13)
    text(s,6.82,4.59,5.43,1.18,'训练过程中观察验证集准确率；\n结果保存在outputs/course_clean/。',19)
    text(s,.9,6.14,11.45,.55,'轮数与batch size没有固定答案，请在报告中写明自己的设置。',17,GRAY)
    # 13. Curves and metrics.
    s=d.slide('12 训练结果分析')
    codebox(s,.82,1.98,11.7,1.25,'from IPython.display import display, Image\ndisplay(Image("outputs/course_clean/curves.png"))',19)
    prompts=[
        ('训练损失','是否总体下降？\n是否出现明显波动？'),
        ('验证结果','准确率怎样变化？\n最佳结果出现在哪一轮？'),
        ('测试结果','加载验证集选出的模型，\n在独立测试集上评估。'),
    ]
    for i,(heading,body) in enumerate(prompts):
        x=.83+i*4.17
        rect(s,x,3.66,3.88,1.9,LIGHT,rounded=True)
        text(s,x+.17,3.86,3.52,.43,heading,22,BLUE,True)
        text(s,x+.17,4.53,3.52,.88,body,18)
    text(s,.93,6.03,11.4,.61,'history.csv保存每轮结果；metrics.json保存最佳轮次、参数和运行时间。',18,GRAY)
    # 14. Evaluation, prediction and errors.
    s=d.slide('13 测试、单图预测与错例检查')
    codebox(s,.8,1.96,11.75,1.62,'!python evaluate_course_corruption.py \\\n  --checkpoint outputs/course_clean/best.pt --output outputs/course_clean_eval\n!python predict.py --checkpoint outputs/course_clean/best.pt --index 0',15)
    text(s,.88,3.91,6.22,1.93,'Clean：基础任务的独立测试结果。\nMedium-Flip / Loss：加分项使用。\n将--index改成其他测试索引，可查看预测概率。\n结合预测图和错误索引分析典型错例。',18)
    picture(s,ROOT/'assets/example_digit.png',8.51,3.75,2.55,2.18)
    text(s,7.62,6.03,4.37,.4,'真实标签7 ｜ 模型预测7',19,BLUE,True,PP_ALIGN.CENTER)
    # 15. Parameter comparison.
    s=d.slide('14 参数对比实验')
    codebox(s,.81,2.0,11.71,2.13,'COMPARISON_EPOCHS = int(input("epochs: "))\nCOMPARISON_BATCH_SIZE = int(input("batch: "))\n!python train_course_subset.py --method clean \\\n  --epochs {COMPARISON_EPOCHS} --batch-size {COMPARISON_BATCH_SIZE} \\\n  --clean-augmentations --output outputs/comparison',14)
    text(s,.88,4.35,5.75,1.63,'轮数和batch size均可自行设置。\n建议一次只改变一个参数，其他条件保持一致。\n保留第一次结果，另存outputs/comparison。',18)
    rect(s,7.02,4.31,5.18,1.72,LIGHT,rounded=True)
    text(s,7.24,4.47,4.74,1.38,'比较内容\n· 验证集准确率与训练时间\n· 损失曲线和收敛速度\n· 参数变化带来的收益与代价',17)
    text(s,.9,6.30,11.35,.4,'报告中列出两次设置及结果，并说明比较结论。',16,GRAY)
    # 16. AutoDL and common problems combined.
    s=d.slide('15 AutoDL运行与常见问题')
    codebox(s,.78,1.94,6.13,3.55,f'cd /root/autodl-tmp\ngit clone {REPO}.git\ncd byteformer-mnist-course\npython -m pip install -r requirements.txt\npython prepare.py\npython train_course_subset.py --method clean \\\n  --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \\\n  --clean-augmentations --output outputs/course_clean',12)
    problems=[('没有使用GPU','检查Kaggle加速器或AutoDL实例'),('缺少Python包','重新安装requirements.txt'),('显存不足','减小batch size并记录实际值'),('输出目录已有文件','改用新的output目录')]
    for i,(heading,body) in enumerate(problems):
        y=2.0+i*.91
        rect(s,7.25,y,5.0,.75,LIGHT,rounded=True)
        text(s,7.43,y+.11,1.78,.3,heading,16,BLUE,True)
        text(s,9.17,y+.11,2.88,.43,body,15)
    text(s,.88,5.80,6.05,.79,'运行前在Terminal中设置EPOCHS和BATCH_SIZE；\n实验结束后下载outputs并在控制台关机。',17,GRAY)
    link(s,7.34,6.00,4.8,'AutoDL官方快速开始','https://www.autodl.com/docs/quick_start/',16)
    # 17. Submission requirements.
    s=d.slide('16 实验结果与报告提交')
    text(s,.86,1.98,6.03,3.71,'① 保存训练曲线和测试结果。\n② 完成一次参数对比。\n③ 展示单图预测或典型错例。\n④ 填写实验环境、命令、参数与结果分析。\n⑤ 打包代码、结果图和实验报告。\n\n数据集和Python环境不需要重复提交。',20)
    checks=['完成训练、验证和测试流程','记录实际轮数与batch size','区分验证集结果和测试集结果','说明曲线变化与参数对比','给出预测结果与错例分析','提交个人实验报告']
    text(s,7.17,2.03,5.0,.47,'基础任务检查',23,BLUE,True)
    for i,item in enumerate(checks):
        y=2.68+i*.51
        rect(s,7.14,y,5.07,.45,LIGHT)
        text(s,7.29,y+.05,4.75,.35,item,17)
    text(s,.9,6.31,11.43,.57,'报告分别列出训练集、验证集和测试集的用途与结果。',17,RED)
    # 18. Bonus corrupted-bitstream task with concrete directions.
    s=d.slide('17 加分项：损坏码流分类','课程提供与干净测试集对应的损坏测试数据，共包含以下两种基本损坏。')
    rect(s,1.05,2.13,5.18,1.50,'F6E5E1',rounded=True)
    text(s,1.24,2.31,4.80,.39,'Bit flip｜比特翻转',21,RED,True,PP_ALIGN.CENTER)
    text(s,1.25,2.86,4.78,.56,'翻转选中字节的1位（长度不变）\n10100110 → 10100010',14,INK,align=PP_ALIGN.CENTER)
    rect(s,7.10,2.13,5.18,1.50,'F4EAD8',rounded=True)
    text(s,7.29,2.31,4.80,.39,'Byte loss｜字节丢失',21,'A46A25',True,PP_ALIGN.CENTER)
    text(s,7.30,2.86,4.78,.56,'删除字节，后续字节前移\nFF D8 A1 3C → FF D8 3C',14,INK,align=PP_ALIGN.CENTER)
    text(s,.83,3.93,2.75,.4,'可以尝试的方向',21,BLUE,True)
    directions=[('① 结果分析','比较Clean、Flip、Loss，观察哪类损坏影响更大。'),('② 损坏增强','训练时随机加入bit flip和byte loss样本。'),('③ 一致性约束','让同一图像的干净码流与损坏码流输出接近。')]
    for i,(heading,body) in enumerate(directions):
        x=.82+i*4.02
        rect(s,x,4.42,3.72,1.18,LIGHT,rounded=True)
        text(s,x+.14,4.57,3.44,.33,heading,17,BLUE,True)
        text(s,x+.14,4.96,3.44,.52,body,14)
    text(s,.86,5.91,2.22,.35,'参考资料：',15,GRAY,True)
    link(s,2.25,5.89,2.55,'CBSU-ALLM','https://doi.org/10.1016/j.patcog.2026.114151',14)
    link(s,4.65,5.89,2.25,'BRACE','https://arxiv.org/abs/2608.15695',14)
    link(s,6.65,5.89,2.25,'BSCV','https://arxiv.org/abs/2309.13890',14)
    text(s,.87,6.37,11.45,.34,'提交改进方法、训练设置以及三类测试结果；可自行组合其他增强或鲁棒训练方法。',15,GRAY)
    # 19. References and entry points.
    s=d.slide('18 配套材料与参考文献','实验步骤、命令和数据说明见课程仓库与Notebook。')
    sources=[
        ('课程仓库：代码、Notebook、数据和报告模板',REPO),
        ('ByteFormer：Bytes Are All You Need','https://arxiv.org/abs/2306.00238'),
        ('MEGABYTE：Predicting Million-byte Sequences','https://arxiv.org/abs/2305.07185'),
        ('MambaByte：Token-free Selective State Space Model','https://arxiv.org/abs/2401.13660'),
        ('bGPT：Byte Models are Digital World Simulators','https://arxiv.org/abs/2402.19155'),
        ('mBLM：Hierarchical Transformers for Byte Modeling','https://arxiv.org/abs/2502.14553'),
        ('Kaggle Notebook官方文档','https://www.kaggle.com/docs/notebooks'),
        ('AutoDL官方快速开始','https://www.autodl.com/docs/quick_start/'),
    ]
    for i,(label,url) in enumerate(sources):link(s,.92,2.02+i*.50,11.55,label,url,16)
    assert len(d.prs.slides)==19
    out=ROOT.parent/('_reference_analysis/course_preview.pptx' if args.draft else 'ByteFormer_MNIST_零基础实验课.pptx')
    out.parent.mkdir(parents=True,exist_ok=True);d.prs.save(out)
    if not args.draft:shutil.copy2(out,ROOT/'docs'/out.name)
    if args.pdf:
        subprocess.run(['libreoffice','-env:UserInstallation=file:///tmp/byteformer_slides_export','--headless','--convert-to','pdf','--outdir',str(out.parent),str(out)],check=True,timeout=180)
        pdf=out.with_suffix('.pdf')
        if not pdf.exists():raise RuntimeError('LibreOffice did not produce PDF')
        if not args.draft:shutil.copy2(pdf,ROOT/'docs'/pdf.name)
    print(f'{len(d.prs.slides)} slides -> {out}')


if __name__=='__main__':main()
