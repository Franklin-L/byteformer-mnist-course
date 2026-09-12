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
        self.prs.core_properties.title='ByteFormer 微调 MNIST：零基础实验课'
        self.prs.core_properties.author='Franklin-L'
        self.prs.core_properties.subject='Kaggle / AutoDL · 真实预训练微调 · 学生操作指南'
    def slide(self,title,subtitle=''):
        s=self.prs.slides.add_slide(self.prs.slide_layouts[6])
        picture(s,ASSETS/'hust_wordmark.png',.55,.2,2.65,.52)
        text(s,8.5,.24,4.2,.35,'ByteFormer · MNIST 实验课',13,GRAY,align=PP_ALIGN.RIGHT)
        text(s,.6,.86,12.1,.56,title,30,INK,True)
        rect(s,.65,1.5,12.02,.023,'BACBD7')
        if subtitle: text(s,.66,1.66,12,.55,subtitle,17,GRAY)
        rect(s,.28,6.97,.32,.32,'B8D7EA'); rect(s,.68,6.97,.32,.32,'C5C3D3')
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
    text(s,2.8,3.03,9.7,1.13,'ByteFormer 微调 MNIST',36,BLUE,True)
    text(s,2.82,4.28,9.5,.75,'零基础图像分类实验课',30,INK)
    text(s,2.83,5.3,9.6,.8,'按步骤运行 · 查看真实结果 · 亲手修改一个参数',20,GRAY)
    link(s,2.85,6.43,10,REPO,REPO,16)
    # 2. Objective and route.
    s=d.bullets('01 任务要求与完成路线',[('完成一次真实微调','加载官方预训练ByteFormer，完成MNIST手写数字0—9分类。'),('亲手完成三个操作','自行设置参数并微调 → 观察验证曲线 → 查看并解释一个错例。'),('交付自己的实验记录','保留自己的指标、曲线、错例和Word报告，说明参数选择与结果。')])
    # 3. ByteFormer and fine tuning, merged.
    s=d.slide('02 实验原理：文件字节输入 + 预训练微调')
    labels=['手写数字图','JPEG 文件','字节序列','ByteFormer','数字0—9']
    for i,label in enumerate(labels):
        x=.72+i*2.48;rect(s,x,2.15,2.16,1.0,LIGHT,rounded=True);text(s,x+.07,2.45,2,.48,label,21,BLUE,True,PP_ALIGN.CENTER)
        if i<4:text(s,x+2.17,2.49,.32,.35,'→',22,BLUE)
    text(s,.86,3.57,11.65,2.35,'① 加载Apple官方ImageNet JPEG预训练ByteFormer Tiny。\n② 保留完整12层主干，把原1000类分类头换成10类。\n③ 使用MNIST继续训练全部参数，输出数字0—9的分类结果。',23)
    text(s,.87,6.07,11.6,.6,'学生只需运行脚本；图像转JPEG、字节读取、模型加载均已封装。',19,GRAY)
    # 4. Explicit disjoint splits.
    s=d.slide('03 MNIST 数据与训练 / 验证 / 测试划分')
    picture(s,digits_asset(),.65,1.85,5.05,2.5)
    text(s,.86,4.69,4.8,1.0,'28×28灰度图；10个数字类别。\n原始文件已随GitHub仓库提供。',20)
    rows=[('训练','50,000张','更新模型参数'),('验证','1,000张','每轮检查，选择best.pt'),('测试','10,000张','模型确定后最终评估')]
    for i,(a,b,c) in enumerate(rows):
        y=2.05+i*1.18;rect(s,6.08,y,6.43,1.0,LIGHT,rounded=True)
        text(s,6.25,y+.10,2.0,.80,a+'\n'+b,20,BLUE,True)
        text(s,8.55,y+.29,3.8,.5,c,20)
    text(s,.88,6.12,11.65,.65,'官方60,000张训练图先分为50,000训练池+10,000验证池；本课固定取1,000验证图，余9,000张未使用。',16,GRAY)
    # 5. Platform choice and material entry.
    s=d.table('04 平台与配套材料：优先 Kaggle，备用 AutoDL',['路线','学生要准备什么','使用方式'],[['Kaggle免费GPU','账号、可用GPU额度、Internet','导入课程Notebook，逐格运行'],['AutoDL备用','按当前报价租1张GPU','选择PyTorch镜像，打开Terminal'],['本地CPU','已有Python/PyTorch环境','小样本流程检查，不替代正式实验']], [2.6,4.5,4.7],subtitle='免费额度、可用GPU与租用价格，以平台当前页面为准。')
    link(s,.84,5.55,5.4,'Kaggle：创建Notebook',KAGGLE,18);link(s,6.75,5.55,5.4,'AutoDL：实例与价格',AUTODL,18)
    text(s,.86,6.13,11.6,.6,'配套：course_kaggle.ipynb逐步操作 ｜ README完整命令 ｜ Word实验报告模板。',18)
    # 6. External platform tutorials and the course entry.
    s=d.slide('05 Kaggle GPU：课前教程与课程入口')
    tutorials=[
        ('知乎｜Kaggle GPU资源使用教程——针对超级小白','https://zhuanlan.zhihu.com/p/18209757723'),
        ('CSDN｜科研小白扫盲：Kaggle平台使用指导指南','https://blog.csdn.net/yyyyyybw/article/details/148336854'),
        ('Kaggle官方｜Notebook使用文档','https://www.kaggle.com/docs/notebooks'),
    ]
    for i,(label,url) in enumerate(tutorials):
        y=2.0+i*.86
        rect(s,.82,y,11.67,.68,LIGHT,rounded=True)
        link(s,1.03,y+.12,11.19,label,url,21)
    link(s,.96,4.88,11.4,'课程Notebook：下载course_kaggle.ipynb',REPO+'/raw/refs/heads/main/course_kaggle.ipynb',21)
    text(s,.99,5.58,11.35,.78,'按教程准备好GPU后，导入课程Notebook，逐格运行。\n环境检查显示 GPU available: True，即可继续实验。',21)
    # 7. Download/prep/files combined.
    s=d.slide('06 获取代码、数据与预训练权重','Kaggle笔记本已写好这些命令，按顺序运行即可。')
    command=f'!git clone {REPO}.git\n%cd /kaggle/working/byteformer-mnist-course\n!python -m pip install -r requirements.txt\n!python prepare.py'
    codebox(s,.8,2.35,11.73,1.92,command,17)
    text(s,.87,4.65,11.6,.8,'看到[READY]即准备完成：MNIST约11.6MB随仓库提供；官方权重约64MB自动下载并校验。',20)
    text(s,.88,5.72,11.6,.92,'prepare.py：准备资源　train.py：微调　evaluate.py：评估　predict.py：预测\ndata/：数据　checkpoints/：预训练权重　outputs/：你自己的实验结果',18,GRAY)
    # 8. Training parameters + command.
    s=d.slide('07 开始训练：看懂参数，运行一条命令')
    settings=[('训练样本','50,000'),('验证 / 测试','1,000 / 10,000'),('训练轮数','自行设置'),('batch size','自行设置，参考32'),('学习率','主干1e-4；分类头10倍')]
    for i,(a,b) in enumerate(settings):
        y=2.03+i*.69;rect(s,.8,y,5.52,.59,LIGHT);text(s,.94,y+.1,2.18,.4,a,19,BLUE,True);text(s,3.15,y+.1,3.04,.43,b,18)
    codebox(s,6.77,2.09,5.72,1.88,'EPOCHS = int(input("epochs: "))\nBATCH_SIZE = int(input("batch: "))\n!python train.py --epochs {EPOCHS} \\\n  --batch-size {BATCH_SIZE}',16)
    text(s,6.91,4.25,5.38,1.6,'看到[DONE]即训练完成。\n结果保存到outputs/baseline/。\n请记录实际使用的参数。',20)
    text(s,.9,6.12,11.45,.55,'训练轮数和batch size均可调整；参考值仅供起步，测试集只作最终评估。',17,GRAY)
    # 9. Read the student's own learning curves and metrics.
    s=d.slide('08 查看自己的训练结果')
    codebox(s,.82,2.0,11.7,1.28,'from IPython.display import display, Image\ndisplay(Image("outputs/baseline/curves.png"))',20)
    prompts=[
        ('训练损失','损失是否逐渐下降？\n后期是否还在改善？'),
        ('验证表现','验证准确率怎样变化？\n与训练表现差距多大？'),
        ('测试结果','加载验证集选出的模型，\n在测试集上评估并记录。'),
    ]
    for i,(heading,body) in enumerate(prompts):
        x=.83+i*4.17
        rect(s,x,3.78,3.88,1.91,LIGHT,rounded=True)
        text(s,x+.17,3.96,3.52,.43,heading,23,BLUE,True)
        text(s,x+.17,4.65,3.52,.92,body,19)
    text(s,.93,6.12,11.4,.55,'指标与参数保存在metrics.json中；结合曲线，解释本次实验的结果。',19,GRAY)
    # 10. Evaluation, inference, and mistakes on one page.
    s=d.slide('09 评估、单图预测与错例检查')
    codebox(s,.8,2.0,11.75,1.31,'!python evaluate.py --checkpoint outputs/baseline/best.pt\n!python predict.py --checkpoint outputs/baseline/best.pt --index 0',17)
    text(s,.88,3.69,6.03,2.16,'评估：读取best.pt，保存evaluation.json。\n预测：显示真实标签与预测，生成单图结果。\n实操：换一个0—9999的测试索引。\n错例：Notebook自动给出错误样本索引。',20)
    picture(s,ROOT/'assets/example_digit.png',8.49,3.55,2.6,2.28)
    text(s,7.5,5.95,4.6,.45,'真实标签7 ｜ 模型预测7',20,BLUE,True,PP_ALIGN.CENTER)
    text(s,.9,6.24,6.5,.47,'Notebook另提供完整预测图和混淆矩阵，供查找与分析错例。',16,GRAY)
    # 11. Student-selected parameters and a compact worksheet.
    s=d.slide('10 学生实操：自行设置，观察验证结果')
    codebox(s,.81,2.03,11.71,1.92,'COMPARISON_EPOCHS = int(input("epochs: "))\nCOMPARISON_BATCH_SIZE = int(input("batch: "))\n!python train.py --epochs {COMPARISON_EPOCHS} \\\n  --batch-size {COMPARISON_BATCH_SIZE} --output outputs/comparison',17)
    text(s,.88,4.18,5.75,1.96,'轮数和batch size均自行设置，\n建议一次只改变其中一项。\n数据、随机种子和预训练来源相同。\n保留baseline，另存comparison。',19)
    rect(s,7.04,4.12,5.16,2.08,LIGHT,rounded=True)
    text(s,7.25,4.25,4.72,1.81,'用自己的结果回答\n· 验证准确率是否提高？\n· 训练损失怎样变化？\n· 改变设置的收益与代价？\n· 为什么还需要验证集？',17)
    text(s,.9,6.37,11.35,.4,'记录两次实验的参数、验证准确率与耗时，解释设置变化带来的影响。',16,GRAY)
    # 12. Entire AutoDL route on one page.
    s=d.slide('11 AutoDL 备用路线：开机 → 运行 → 下载 → 关机')
    text(s,.86,1.96,11.65,.68,'租用1张GPU并选择PyTorch镜像 → 打开JupyterLab → Terminal；以下命令前不加 !。',20)
    codebox(s,.82,2.65,11.71,2.73,f'cd /root/autodl-tmp\ngit clone {REPO}.git\ncd byteformer-mnist-course\npython -m pip install -r requirements.txt\npython prepare.py\nread -p "epochs: " EPOCHS\nread -p "batch size (ref 32): " BATCH_SIZE\npython train.py --epochs "$EPOCHS" --batch-size "$BATCH_SIZE"',16)
    text(s,.91,5.56,11.43,.89,'运行后下载outputs中的指标和图；如需继续预测，保存best.pt。\n结束后回控制台关机：关闭浏览器不等于关机。价格以页面当前报价为准。',19)
    link(s,.9,6.57,11.3,'AutoDL官方快速开始（实例、JupyterLab、Terminal）','https://www.autodl.com/docs/quick_start/',14)
    # 13. FAQ.
    s=d.table('12 常见问题：从错误最后一行定位',['现象','处理'],[['GPU available: False','开启Kaggle GPU；或确认AutoDL实例和PyTorch环境'],['No module named ...','在当前课程目录安装requirements.txt，确认使用同一环境'],['下载超时 / 校验失败','开启Internet重试；或使用教师预下载资源包'],['CUDA out of memory','减小batch size，如32降到16或8，并记录实际值'],['输出目录已存在 / best.pt找不到','重跑使用新output；评估checkpoint对应实际训练目录'],['验证表现不再改善','查看训练与验证曲线，检查样本量、预训练权重与学习率']], [4.15,7.65])
    text(s,.88,6.53,11.45,.32,'CPU备用：README提供小样本流程检查命令，便于先熟悉代码运行。',14,GRAY)
    # 14. Packaging and grading combined.
    s=d.slide('13 下载结果、填写报告并提交')
    text(s,.86,1.98,6.03,3.64,'① 运行Notebook最后的打包单元格。\n② 在文件面板下载byteformer_mnist_results.zip。\n③ 用Word模板填写自己的环境、命令、指标和解释。\n④ 附两组曲线、预测图、错例证据。\n\n默认不用提交数据集、Python环境或大模型。',21)
    checks=['训练、验证和测试流程完整','记录实际参数与运行环境','说明训练与验证曲线变化','完成一次自选参数对比','查看预测并分析错例','提交自己的实验报告']
    text(s,7.17,2.03,5.0,.47,'基础任务达标检查',23,BLUE,True)
    for i,item in enumerate(checks):
        y=2.68+i*.51
        rect(s,7.14,y,5.07,.45,LIGHT)
        text(s,7.29,y+.05,4.75,.35,item,17)
    text(s,.9,6.31,11.43,.57,'提交渠道与截止时间由任课教师说明；只填写真实运行结果，验证与测试指标不要混写。',17,RED)
    # 15. Optional higher-grade tasks; no additional experiments are required here.
    s=d.slide('14 拓展加分任务')
    rect(s,.83,1.96,11.68,.86,LIGHT,rounded=True)
    text(s,1.03,2.17,11.25,.46,'完成MNIST规定任务并提交实验报告，即达标及格。',23,BLUE,True)
    tasks=[
        (.83,'加分任务一｜CIFAR-10微调','把ByteFormer微调流程迁移到CIFAR-10。\n自行适配数据读取与分类任务，\n完成训练、验证和测试。'),
        (6.81,'进阶加分任务二｜Stanford40','完成斯坦福40动作识别数据集的微调分类。\n自行准备数据、调整分类头并设置参数，\n分析预测结果与典型错例。'),
    ]
    for x,heading,body in tasks:
        rect(s,x,3.22,5.7,2.42,LIGHT,rounded=True)
        text(s,x+.17,3.44,5.34,.46,heading,22,BLUE,True)
        text(s,x+.17,4.18,5.34,1.28,body,18)
    text(s,.98,5.99,11.35,.84,'完成额外任务可获得加分，挑战进阶任务可争取更高分。\n附上代码、数据划分、实际结果与简要分析，说明你做了哪些调整。',20)
    # 16. References and concrete entry.
    s=d.slide('15 课程入口与参考资料','详细点击步骤、完整命令和排错说明，统一放在配套指南与Notebook。')
    sources=[('课程仓库：代码、Notebook、数据、PPT、报告模板',REPO),('Kaggle Notebook 官方文档','https://www.kaggle.com/docs/notebooks'),('AutoDL 官方快速开始','https://www.autodl.com/docs/quick_start/'),('ByteFormer 原论文：Bytes Are All You Need','https://arxiv.org/abs/2306.00238'),('Apple CoreNet：ByteFormer代码与预训练权重','https://github.com/apple/corenet/tree/main/projects/byteformer'),('MNIST：CVDF镜像与原作者说明','https://github.com/cvdfoundation/mnist')]
    for i,(label,url) in enumerate(sources):link(s,.92,2.25+i*.59,11.55,label,url,19)
    assert len(d.prs.slides)==16
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
