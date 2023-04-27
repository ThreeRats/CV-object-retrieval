from tkinter import *
from PIL import ImageTk, Image
from search import *
import argparse

photo = None
img = None
dataset = None

def preshow() -> None:
    """
    预览选中的图片
    
    return:None
    """

    global photo
    global img
    global dataset
    
    path = entry.get() # 获取输入框中的图片路径

    # 使用PIL库加载图片并展示
    try:
        img = Image.open(dataset + '/' + path + '.jpg')
        img = img.resize((400, 400), Image.ANTIALIAS)
        newwindow = Toplevel(root)
        newwindow.title("预览图片")
        newwindow.geometry('400x400')
        photo = ImageTk.PhotoImage(img)
        label = Label(newwindow, image=photo)
        label.pack(side='bottom', fill='both', expand='yes')

    except FileNotFoundError:
        newwindow = Toplevel(root)
        newwindow.title("出错了")
        newwindow.geometry('200x200')
        label = Label(newwindow, text='图片不存在！', font=('Helvetica', 18))
        label.pack()

    except:
        pass

def show() -> None:
    """
    展示最相似的5张图片

    return: None
    """

    global dataset

    query_name = entry.get()

    try:
        similar_names = search(query_name)
        # similar_names = ['all_souls_000000', 'all_souls_000001', 'all_souls_000002', 'all_souls_000003', 'all_souls_000005']
        images = [ImageTk.PhotoImage(Image.open( dataset + f"/{name}.jpg" ).resize((200,200)))
                  for name in similar_names]


        newwindow = Toplevel(root)
        newwindow.title("最相似的5张图片")
        newwindow.geometry('1000x1000')
        
        for _, img in enumerate(images):
            label = Label(newwindow, image=img)
            label.image = img
            label.pack(side="left")

    except FileNotFoundError:
        newwindow = Toplevel(root)
        newwindow.title("出错了")
        newwindow.geometry('200x200')
        label = Label(newwindow, text='图片不存在！', font=('Helvetica', 18))
        label.pack()

    except:
        pass



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="选择使用的图片数据集")

    args = parser.parse_args()
    
    dataset = args.dataset

    root = Tk()
    root.title("图片检索系统")
    root.geometry('400x400')

    title = Label(root, text='欢迎来到图片检索系统！', font=('Helvetica', 18))
    title.pack(pady=10)

    info = Label(root, text='开发者：何奕骁 杜嘉骏 官子鸣', font=('Helvetica', 18))
    info.pack(pady=10)

    # 创建一个Frame框架来放置图片
    frame = Frame(root)
    frame.pack()

    # 创建一个输入框和搜索按钮
    entry = Entry(root, width=150, font=("Helvetica", 18))
    entry.pack(pady=10)

    button = Button(root, text="预览",font=('Helvetica', 18), command=preshow)
    button.pack(side='left', padx=15, pady=10, expand=True)

    button = Button(root, text="检索",font=('Helvetica', 18), command=show)
    button.pack(side='left', padx=15, pady=10, expand=True)

    button = Button(root, text="退出", font=('Helvetica', 18), command=root.destroy)
    button.pack(side='right', padx=15, pady=10, expand=True)



    root.mainloop()