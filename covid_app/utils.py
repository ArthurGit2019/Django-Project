import matplotlib.pyplot as plt
import base64
from io import BytesIO

def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph

def get_scatter_plot(x, y):
    plt.switch_backend('AGG')
    plt.figure(figsize=(10,5))
    plt.title('Deaths by Confirmed Cases')
    plt.scatter(x,y)
    plt.xlabel('Confirmed Cases')
    plt.ylabel('Deaths')
    graph = get_graph()
    return graph

def get_line_plot(x, y):
    plt.switch_backend('AGG')
    plt.figure(figsize=(10,5))
    plt.title('Deaths by Confirmed Cases')
    plt.plot(x,y)
    plt.xlabel('Confirmed Cases')
    plt.ylabel('Deaths')
    graph = get_graph()
    return graph