from domino import *
import definitions
import matplotlib.pyplot as plt

EPISODES = 50

juego = Juego(6,4,True)

E = []
R = []
loss = []
juego.policy.load_Model( 'models/supervisado_mejorado.h5' )
open("loss.txt","w").close()


for episode in range(EPISODES):
    print(f'Partida {episode+1:d}/{EPISODES:d}...')
    juego.jugar()

    juego.reset()
#juego.policy.saveModel("supervisado_1")

total_buenas=0
total_en_mano=0
total_no_mano=0
total_total=0
for jug in juego.jugadores:
    total_buenas+=jug.jugadas_buenas
    total_en_mano+=jug.jugadas_mano
    total_no_mano+=jug.jugada_NM
    total_total+=jug.jugadas_totales

print( f'Hizo {total_buenas:d} buenas jugadas de {total_total:d} jugadas totales Accuracy: {100*total_buenas/total_total:.2f}.\
    Jugadas que servían pero no tenía en mano {total_en_mano:d} porcentaje:{total_en_mano/total_total:0.2f}.Jugadas que eran válidas pero no las tenia o no le atino al lado {total_no_mano:d} porcentaje {total_no_mano/total_total:0.2f} ' )



'''
plt.figure()
plt.scatter( E, R )
plt.xlabel("Episodes")
plt.ylabel("Rewards")


plt.figure()
plt.hist( R )
plt.xlabel("Rewards")'''

plt.figure()
plt.plot( juego.policy.loss_history )

plt.show()