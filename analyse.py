import numpy as np

from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# "d" is the space_headway, "ds" is the relative speed, "s" is the speed,

class Analyse:

    def __init__(self, vehicle, data_plot):
        self.vehicle = vehicle
        self.data_plot = data_plot

        self.rangeStep = 1000
        self.collisions_table = np.array(np.zeros([int(self.data_plot.get("steps")/self.rangeStep)]))
        for collision in self.data_plot.get("collisions"):
            self.collisions_table[int(collision/self.rangeStep)] += 1
        
    def plot_all_result(self): #Plot all results after simulation running
        
        q_3d = True

        if q_3d:
            self.plot_qTable_3d_()

        self.plot_spaceHeadway()
        self.plot_speed()
        self.plot_relativeSpeed()
        self.plot_collisions()
        self.show_tables_()
        
    def plot_qTable_3d_(self): #plot the qTable on 3D
       #Markers of the plot
        m = ['*', '+', 'x']
        c = ['r', 'y', 'g']

        plt.ion()
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for d in self.vehicle.index_space_headway:
            i_d = self.vehicle.index_space_headway.get(d)
            for ds in self.vehicle.index_relative_speed:
                i_ds = self.vehicle.index_relative_speed.get(ds)
                for s in self.vehicle.index_speed:
                    i_s = self.vehicle.index_speed.get(s)
                    a = int(np.argmax(self.vehicle.q[i_d, i_ds, i_s]))
                    if not (a == 0 and self.vehicle.q[i_d, i_ds, i_s, a] == 0):
                        ax.scatter(d, round(ds * 3.6, 0), round(s * 3.6, 0), marker=m[a], c=c[a])

        ax.set_xlabel('Space headway (m)')
        ax.set_ylabel('Relative Speed (km/h)')
        ax.set_zlabel('Speed (km/h)')
        ax.set_title("Q-Table group by acceleration")
        ax.legend(handles=[patches.Patch(color='red', label='↑'),
                           patches.Patch(color='blue', label='↓'),
                           patches.Patch(color='green', label='=')],
                  loc=2)

        plt.show()
        plt.draw()
        plt.savefig("result/3DQtable.png")
        plt.pause(0.001)

    def plot_collisions(self): #Plot Collisions over time
        
        plt.ion()
        plt.show()
        plt.style.use('seaborn')

        fig = plt.figure(figsize=(12, 6))
        fig.suptitle('Collision over time')
    
        ax = fig.add_subplot(111)

        ax.bar(np.linspace(0, self.data_plot.get("steps"), len(self.collisions_table)), self.collisions_table, width=1.0, facecolor='b', edgecolor='g')
        ax.plot(np.linspace(0, self.data_plot.get("steps"), len(self.collisions_table)), self.collisions_table, 'k--', alpha=0.5)
        ax.set_ylabel('Collisions')
        ax.set_title("Collisions per 1000 steps", fontsize=10)

        plt.show()
        plt.draw()
        plt.savefig('result/collisions.png')
        plt.pause(0.001)

    def plot_spaceHeadway(self): #Plot the space headway over time
        
        plt.ion()
        plt.show()
        plt.style.use('seaborn')

        fig = plt.figure(figsize=(12, 6))
        fig.suptitle('Space headway over time')
    
        ax = fig.add_subplot(111)

        ax.plot(np.linspace(0, self.data_plot.get("steps"), len(self.data_plot.get("space_headway"))),
                   self.data_plot.get("space_headway"), 'k', alpha=0.85)
        ax.set_ylabel('Space headway (m)')
        ax.set_title("Space headway (m) over time", fontsize=10)
        plt.show()
        plt.draw()
        plt.savefig('result/spaceHeadway.png')
        plt.pause(0.001) 
        
    def plot_relativeSpeed(self): #Plot relative speed over time
       
        plt.ion()
        plt.show()
        plt.style.use('seaborn')

        fig = plt.figure(figsize=(12, 6))
        fig.suptitle('Relative speed over time')
    
        ax = fig.add_subplot(111)

        ax.plot(np.linspace(0, self.data_plot.get("steps"), len(self.data_plot.get("relative_speed"))),
                   self.data_plot.get("relative_speed"), 'k', alpha=0.85)
        ax.set_xlabel('Simulation steps')
        ax.set_ylabel('Relative speed (km/h)')
        ax.set_title("Relative speed (km/h) over time", fontsize=10)

        plt.show()
        plt.draw()
        plt.savefig('result/relativeSpeed.png')
        plt.pause(0.001)             

    def plot_speed(self): #Plot spedd over time
        
        plt.ion()
        plt.show()
        plt.style.use('seaborn')

        fig = plt.figure(figsize=(12, 6))
        fig.suptitle('Speed over time')
    
        ax = fig.add_subplot(111)

        ax.plot(np.linspace(0, self.data_plot.get("steps"), len(self.data_plot.get("speed"))),
                   self.data_plot.get("speed"), 'k', alpha=0.85)
        ax.set_xlabel('Simulation steps')
        ax.set_ylabel('Speed (km/h)')
        ax.set_title("Speed (km/h) over time", fontsize=10)

        plt.show()
        plt.draw()
        plt.savefig('result/speed.png')
        plt.pause(0.001)             
 
    def show_tables_(self): #Show the table that describe the steps, the mean and the standard deviation
        plt.ion()
        plt.show()
        plt.style.use('seaborn')

        fig = plt.figure(figsize=(12, 6))
        fig.suptitle('Collision and state')

        ax = []

        n_rows = 1
        n_cols = 4

        for i in range(4):
            ax.append(fig.add_subplot(n_rows, n_cols, i+1))

        cell_text = []
        total = 0
        for it, b in enumerate(self.collisions_table):
            total += b
            if it < 10 or (it < 50 and (it + 1) % 5 == 0) or (it < 100 and (it + 1) % 10 == 0):
                cell_text.append([(it+1) * self.rangeStep, int(b), int(total)])

        ax[3].axis('tight')
        ax[3].axis('off')
        ax[3].table(cellText=cell_text, colLabels=("Steps", "Collisions", "Total"), loc='center')
        ax[3].set_title("Collisions", fontsize=10)

        cell_text = []
        space_headway_chunks = [self.data_plot.get("space_headway")[i:i + self.rangeStep]
                                for i in range(0, len(self.data_plot.get("space_headway")), self.rangeStep)]

        for it, chunk in enumerate(space_headway_chunks):
            if it < 10 or (it < 50 and (it + 1) % 5 == 0) or (it < 100 and (it + 1) % 10 == 0):
                cell_text.append([(it+1) * self.rangeStep, round(float(np.mean(chunk)), 2), round(float(np.std(chunk)), 2)])

        ax[2].axis('tight')
        ax[2].axis('off')
        ax[2].table(cellText=cell_text, colLabels=("Steps", "Mean", "St deviation"), loc='center')
        ax[2].set_title("spaceHeadway resume(m)", fontsize=10)

        cell_text = []
        relative_speed_chunks = [self.data_plot.get("relative_speed")[i:i + self.rangeStep]
                                 for i in range(0, len(self.data_plot.get("relative_speed")), self.rangeStep)]

        for it, chunk in enumerate(relative_speed_chunks):
            if it < 10 or (it < 50 and (it + 1) % 5 == 0) or (it < 100 and (it + 1) % 10 == 0):
                cell_text.append([(it+1) * self.rangeStep, round(float(np.mean(chunk)), 2), round(float(np.std(chunk)), 2)])

        ax[1].axis('tight')
        ax[1].axis('off')
        ax[1].table(cellText=cell_text, colLabels=("Steps", "Mean", "St deviation"), loc='center')
        ax[1].set_title("Relative speed resume(km/h)", fontsize=10)

        cell_text = []
        speed_chunks = [self.data_plot.get("speed")[i:i + self.rangeStep]
                        for i in range(0, len(self.data_plot.get("speed")), self.rangeStep)]

        for it, chunk in enumerate(speed_chunks):
            if it < 10 or (it < 50 and (it + 1) % 5 == 0) or (it < 100 and (it + 1) % 10 == 0):
                cell_text.append([(it+1) * self.rangeStep, round(float(np.mean(chunk)), 2), round(float(np.std(chunk)), 2)])

        ax[0].axis('tight')
        ax[0].axis('off')
        ax[0].table(cellText=cell_text, colLabels=("Steps", "Mean", "St deviation"), loc='center')
        ax[0].set_title("Agent speed resume(km/h)", fontsize=10)

        plt.show()
        plt.draw()
        plt.savefig('result/eventSummary.png')
        plt.pause(0.001)
