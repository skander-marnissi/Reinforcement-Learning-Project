import sys
from vehicle import Vehicle


#Inputs options in console 
if len(sys.argv) == 3 or len(sys.argv) == 5:
    if (sys.argv[1] == "-nv" or sys.argv[1] == "--vehicle") and sys.argv[2].isdigit() and 3 <= int(sys.argv[2]) <= 50:
        vehicle = int(sys.argv[2])
    if (sys.argv[1] == "-t" or sys.argv[1] == "--totalduration") and sys.argv[2].isdigit() and 1 <= int(sys.argv[2]) <= 10:
        totalDuration = int(sys.argv[2])
    if len(sys.argv) == 5:
        if (sys.argv[3] == "-nv" or sys.argv[3] == "--vehicle") and sys.argv[4].isdigit() and 3 <= int(sys.argv[4]) <= 50:
            vehicle = int(sys.argv[4])
        if (sys.argv[3] == "-t" or sys.argv[3] == "--totalduration") and sys.argv[4].isdigit() and 1 <= int(sys.argv[4]) <= 10:
            totalDuration = int(sys.argv[4])

#Default duration and vehicles
totalDuration = 5
vehicle = 50


if __name__ == "__main__":

    sys.argv = [sys.argv[0]]

    print(f"{vehicle} vehicles")

    print(f"{totalDuration} * 10000 steps")

    myAgent = Vehicle('sumo/network.net.xml', vehicle)

    myAgent.qlearning(totalDuration)

    myAgent.close_simulation()

