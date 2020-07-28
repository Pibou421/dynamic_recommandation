import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots()
most_common_names = np.load("MostCommonNames.npy").tolist()
most_common_movies = np.load("MostCommonMovies.npy").tolist()


'''most_common_movies = [314,
 277,
 257,
 510,
 1938,
 224,
 418,
 97,
 507,
 461,
 2224,
 0,
 897,
 46,
 2144,
 43,
 615,
 123,
 899,
 3633,
 910,
 659,
 398,
 509,
 1502]
most_common_names = ['Forrest Gump (1994)',
 'Shawshank Redemption, The (1994)',
 'Pulp Fiction (1994)',
 'Silence of the Lambs, The (1991)',
 'Matrix, The (1999)',
 'Star Wars: Episode IV - A New Hope (1977)',
 'Jurassic Park (1993)',
 'Braveheart (1995)',
 'Terminator 2: Judgment Day (1991)',
 "Schindler's List (1993)",
 'Fight Club (1999)',
 'Toy Story (1995)',
 'Star Wars: Episode V - The Empire Strikes Back (1980)',
 'Usual Suspects, The (1995)',
 'American Beauty (1999)',
 'Seven (a.k.a. Se7en) (1995)',
 'Independence Day (a.k.a. ID4) (1996)',
 'Apollo 13 (1995)',
 'Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)',
 'Lord of the Rings: The Fellowship of the Ring, The (2001)',
 'Star Wars: Episode VI - Return of the Jedi (1983)',
 'Godfather, The (1972)',
 'Fugitive, The (1993)',
 'Batman (1989)',
 'Saving Private Ryan (1998)']'''


def display_title(x, y):
    plt.annotate(title_of_point[(x, y)], xy=(x, y), xycoords='data',
        xytext=(-30, -30), textcoords='offset points',
        arrowprops=dict(arrowstyle="->",
                        connectionstyle="arc3,rad=.2")
        )




def onpick(event):
    ind = most_common_movies[event.ind[0]]
    x, y = X_to_scatter[most_common_movies.index(ind)], Y_to_scatter[most_common_movies.index(ind)]
    print(x,y)
    display_title(x, y)

def load_points(X,Y, fig,ax,marker='.', color='#FF6A00', titles=None):
    ax.scatter(X,Y, marker=marker, c=color, picker=True)


V = np.load("Vmatrix.npy")



X_to_scatter, Y_to_scatter = V[most_common_movies,0],V[most_common_movies,1]


#plt.scatter(X_to_scatter,Y_to_scatter)

title_of_point = dict([(tuple([X_to_scatter[i], Y_to_scatter[i]]), most_common_names[i]) for i in range(len(most_common_movies))])

load_points(X_to_scatter, Y_to_scatter, fig, ax)

fig.canvas.mpl_connect('pick_event', onpick)
plt.show()
