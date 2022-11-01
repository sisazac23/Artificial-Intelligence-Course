import plotly_express as px

def plot3D(df,X,Y,Z,color):
    return px.scatter_3d(df,x=X,y=Y,z=Z,color=color,opacity=0.5)