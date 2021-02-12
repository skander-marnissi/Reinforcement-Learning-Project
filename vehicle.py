from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import random
import time
import os
import sys
import optparse

from analyse import Analyse
from sumolib import checkBinary, net 
import traci

class Vehicle:

    def __init__(self, environment , cars_number):
        
        #Initialise the enviroment from traci 
        self.environment = net.readNet(environment)
        self.nbrCars = cars_number
        self.myControlledCarId = "my_car"
        

        self.generate_network_file()
        options = self.get_options()
        params = self.set_parameters(options)
        traci.start(params)
        
        self.totalRewards = 1
        

        # Parameters initialisation of Alpha, Gamma and Epsilon
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 0.1

        # Parameters initialisation of the State
        self.space_headway = {
            "min": 0.,
            "max": 150.,
            "decimals": 0,
            "nb_values": 6
        }

        self.relative_speed = {
            "min": -8.33,
            "max": 8.33,
            "decimals": 2,
            "nb_values": 6
        }

        self.speed = {
            "min": 0.,
            "max": 13.89,
            "decimals": 2,
            "nb_values": 6
        }

        #Actions to take 1: accelerate 2: decelerate 0: nothing
        self.action = [1, -1, 0]

       # Indexes from Traci  
        self.index_space_headway = \
            dict(((round(i, self.space_headway.get('decimals'))), iteration)
                 for iteration, i in
                 enumerate(np.linspace(self.space_headway.get('min'),
                                       self.space_headway.get('max'),
                                       self.space_headway.get('nb_values'))))

        self.index_relative_speed = \
            dict(((round(i, self.relative_speed.get('decimals'))), iteration)
                 for iteration, i in
                 enumerate(np.linspace(self.relative_speed.get('min'),
                                       self.relative_speed.get('max'),
                                       self.relative_speed.get('nb_values'))))

        self.index_speed = \
            dict(((round(i, self.speed.get('decimals'))), iteration)
                 for iteration, i in
                 enumerate(np.linspace(self.speed.get('min'),
                                       self.speed.get('max'),
                                       self.speed.get('nb_values'))))
    
        self.index_action = \
            dict((i, iteration)
                 for iteration, i in enumerate(self.action))

        # 4 dimentionnal Q-table initialisation
        self.q = np.array(np.zeros([len(self.index_space_headway),
                                    len(self.index_relative_speed),
                                    len(self.index_speed),
                                    len(self.index_action)]))

    def update_speed(self, a, speed): #Update speed of the agent
        print(f"action : {a}")
        return np.max([self.speed.get('min'), speed + a])

    def epsilon_period(self, step): #Optimise epsilone each 1000 steps to make equilibrium between exploration and exploitation
        # Each 1000 steps we update espilon 
        if step % 1000 == 0:
            # We will increase the exploration probabilty
            self.epsilon = round(self.epsilon * 0.9, 6)

    def Egreedy_policy(self, d_t, ds_t, s_t): #Retrun an action to take : either the agent discover or exploit the q-table
        if random.uniform(0, 1) < self.epsilon:
            #Choose action between [1, -1, 0]
            return random.randint(0, 2)
        else:
            #We will choose the optimal action in Q-table
            return np.argmax(self.q[
                self.index_space_headway.get(d_t),
                self.index_relative_speed.get(ds_t),
                self.index_speed.get(s_t)])

    def arrange(self, value, index_dict): #Return the index of discret interval of our agent from the envirement
        index_lower, index_max, index_min = None, np.NINF, np.inf #none -00 +00

        for i in index_dict:
            if i > index_max:
                index_max = i
            if i < index_min:
                index_min = i
            if value >= i:
                index_lower = i
            if index_min <= value < i:
                if (value - index_lower) >= (i - index_lower) / 2:
                    value = i
                else:
                    value = index_lower
                break

        return np.min([index_max, np.max([index_min, value])])

    def qlearning(self, eps): #The Q-learning algorithm to train our agent
        self.set_velocity_mode(self.myControlledCarId, 0)
        

        reward_type = "security_distance"
        speedLimit = True
        state = None
        steps = 0

        plottingList = {
            "collisions": [], "space_headway": [], "relative_speed": [], "speed": [], "steps": 0
        }

        while True:
            print(state)
            if state:
                plottingList["space_headway"].append(state.get("space_headway"))
                plottingList["relative_speed"].append(round(state.get("relative_speed") * 3.6, 0)) #Metre par seconde from traci to km par seconde
                plottingList["speed"].append(round(state.get("speed") * 3.6, 0)) #Metre par seconde from traci to km par seconde

                d_t, ds_t, s_t = \
                    self.arrange(state.get('space_headway'), self.index_space_headway), \
                    self.arrange(state.get('relative_speed'), self.index_relative_speed), \
                    self.arrange(state.get('speed'), self.index_speed)

                a = self.Egreedy_policy(d_t, ds_t, s_t)

                q_t = self.q[
                    self.index_space_headway.get(d_t),
                    self.index_relative_speed.get(ds_t),
                    self.index_speed.get(s_t),
                    self.index_action.get(self.action[a])]

                #Get the updated speed
                update_speed = self.update_speed(self.action[a], state.get('speed'))
                
                #Update agent speed ,step from traci
                self.set_velocity(self.myControlledCarId, update_speed)

                #Simulate new step with traci
                self.generate_simulation_step()
                
                next_state = self.get_state(self.myControlledCarId)

                q_t_m = None
                
                #Check if our agent has been collided with an other vehicle
                if self.isCollided(self.myControlledCarId):
                    #Affect reward -10
                    self.set_reward_after_collision(reward_type) #reward_type "collision"
                    #Set agent speed to 0 
                    self.set_velocity(self.myControlledCarId, 0)
                    q_t_m = 0
                    state = None
                    plottingList["collisions"].append(steps)

                elif next_state:
                 
                    if reward_type == "security_distance":
                        self.set_reward_security_dist_volicity(next_state.get('space_headway'), next_state.get('speed'), speedLimit)

                    print(f" total reward :  {self.totalRewards}")

                    d_t1, ds_t1, s_t1 = \
                        self.arrange(next_state.get('space_headway'), self.index_space_headway), \
                        self.arrange(next_state.get('relative_speed'), self.index_relative_speed), \
                        self.arrange(next_state.get('speed'), self.index_speed)

                    q_t_m = np.max(self.q[
                                          self.index_space_headway.get(d_t1),
                                          self.index_relative_speed.get(ds_t1),
                                          self.index_speed.get(s_t1)])

                    state = next_state

                if q_t_m is not None:
                    self.q[
                        self.index_space_headway.get(d_t),
                        self.index_relative_speed.get(ds_t),
                        self.index_speed.get(s_t),
                        self.index_action.get(self.action[a])] = \
                        (1 - self.alpha) * q_t + self.alpha * (self.totalRewards + self.gamma * q_t_m)

                    
                    print(f"qlearning: {self.q[self.index_space_headway.get(d_t), self.index_relative_speed.get(ds_t), self.index_speed.get(s_t)]}")

                steps += 1
                self.epsilon_period(steps)
            
            else:
                self.generate_simulation_step()
                state = self.get_state(self.myControlledCarId)
                self.set_velocity(self.myControlledCarId, 0)

            if steps > (eps * 10000):
                time.sleep(.1)

            if steps == eps * 10000:
                plottingList["steps"] = steps
                plotting = Analyse(self, plottingList)
                plotting.plot_all_result()

    def set_velocity_mode(self, carId, mode): #Set the velocity-mode from Traci
        #Set vehicle speed mode from traci
        traci.vehicle.setSpeedMode(carId, mode)

    def set_velocity(self, car_id, speed): #Set the velocity from Traci
        #Set vehicle speed from traci
        traci.vehicle.setSpeed(car_id, speed)

    def get_position(self, car_id): #Get the position of our agent from Traci
        #Get agent position from Traci
        return traci.vehicle.getPosition(car_id)

    def get_distance(self, position_1, position_2): #Get the distance between two cars from Traci
        #Get the distance between our agent and another cars from Traci
        return np.sqrt(np.power(position_1[0] - position_2[0], 2)
                       + np.power(position_1[1] - position_2[1], 2))

    def get_carLeader_on_edge(self, car_id): #Get the first car that lead the lane from Traci
        #Get the frist car that lead the lane from Traci
        return traci.vehicle.getLeader(car_id, dist=0.0)

    def get_next_vehicle(self, car_id, border): #Get the first car next to our agent from Traci
        follower_dist = traci.vehicle.getFollower(car_id, dist=0.0)[1]
        return np.min([round(follower_dist, 0), border]) if follower_dist > 0 else border

    def get_edge_id(self, car_id): #Get Edge Id from Traci
        return traci.vehicle.getRoadID(car_id)

    def get_next_edge(self, edge_id): #Get next edge from Traci
        
        #Get the outgoing edges from Traci with the given edge_id
        outgoing_edges = list(self.environment.getEdge(edge_id).getOutgoing().keys())[0]
        return {
            "id": outgoing_edges.getID(),
            "from": outgoing_edges.getFromNode().getID(),
            "to": outgoing_edges.getToNode().getID()
        }

    def get_carsNumber_on_edge(self, edge_id): #Get cars number from Traci
        try:
            return traci.edge.getLastStepVehicleNumber(edge_id)
        except traci.exceptions.TraCIException as e:
            print(e)
            return None

    def get_edges_until_leader(self, car_id):  #Optionnal : get edges from Traci
        e, next_edge, is_car_on_next_edge, edges = None, None, None, []
        e = self.get_edge_id(car_id)
        if e and not e[0] == ':':
            next_edge = self.get_next_edge(e)
            while not is_car_on_next_edge:
                edges.append(next_edge)
                is_car_on_next_edge = self.get_carsNumber_on_edge(next_edge.get("id"))
                next_edge = self.get_next_edge(next_edge.get("id"))
        return edges

    def get_cars_on_edge(self, edge_id): # Get cars of a given edge from Traci
        try:
            return traci.edge.getLastStepVehicleIDs(edge_id)
        except traci.exceptions.TraCIException as e:
            print(e)
            return None

    def get_carLeader_next_edge(self, edges): # Get the next car leader on the next edge from Traci
        if edges:
            return self.get_cars_on_edge(edges[-1].get("id"))[0]
        else:
            return None

    def get_node_coordinates(self, node_id): # Get node coordinates from Traci with the given node id
        return traci.junction.getPosition(node_id)

    def get_carLeader_on_next_edge_distance(self, firstCar_id, secondCar_id, edges): #Get the distance of the next car leader from the next edge with Traci
        if firstCar_id and secondCar_id and edges:
            distance = self.get_distance(self.get_position(firstCar_id),
                                         self.get_node_coordinates(edges[0].get("from")))
            for i in range(len(edges)):
                if i < len(edges) - 1:
                    distance += self.get_distance(self.get_node_coordinates(edges[i].get("from")),
                                                  self.get_node_coordinates(edges[i].get("to")))
                else:
                    distance += self.get_distance(self.get_node_coordinates(edges[i].get("from")),
                                                  self.get_position(secondCar_id))
            return distance
        else:
            return None

    def get_carLeader(self, car_id): #Get the car leader from the actual node from Traci
        leader = {'id': None, 'distance': None}
        leader_on_edge = self.get_carLeader_on_edge(car_id)
        if leader_on_edge:
            leader['id'] = leader_on_edge[0]
            leader['distance'] = leader_on_edge[1]
        else:
            edges = self.get_edges_until_leader(car_id)
            leader['id'] = self.get_carLeader_next_edge(edges)
            leader['distance'] = self.get_carLeader_on_next_edge_distance(car_id, leader.get('id'), edges)
        return leader

    def get_velocity(self, car_id): # Get the velocity of the agent from Traci
        return traci.vehicle.getSpeed(car_id)

    def get_relative_speed(self, firstCar_id, secondCar_id): #Get the relative speed between two given cars from Traci
        return self.get_velocity(firstCar_id) - self.get_velocity(secondCar_id)

    def get_state(self, car_id): #Get the agent states from the envirement from Traci
        carLeader = self.get_carLeader(car_id)
        if carLeader.get('id') and carLeader.get('distance'):
            return {
                "speed": round(self.get_velocity(car_id), 2),
                "relative_speed": round(self.get_relative_speed(car_id, carLeader.get('id')), 2),
                "space_headway": round(carLeader.get('distance'), 0)
            }
        else:
            return None

    def get_accidents(self): #Get the actual accident from Traci
        return traci.simulation.getCollidingVehiclesIDList()

    def isCollided(self, car_id): #Check if our agent collided with another car from Traci
        return car_id in self.get_accidents()

    def get_time(self): # Get the simulation time from Traci
        return traci.simulation.getCurrentTime()

    def resetAll(self): # Reset all 
        pass

    def generate_simulation_step(self): #Generate another a new step from Traci with SUMO simulator
        traci.simulationStep()

    def close_simulation(self): #Close SUMO simulator from Traci
        traci.close()
        sys.stdout.flush()

    def generate_network_file(self): #Generate the network.rou.xml to create the enviromment
        with open("sumo/network.rou.xml", "w") as routes:
            print('<routes>', file=routes)
            print(' <vType accel="2.9" decel="7.5" id="npc_car" type="passenger" length="4.3" minGap="30" maxSpeed="13.89" sigma="0.5" />', file=routes)
            print(' <vType accel="2.9" decel="7.5" id="ai_car" type="passenger" length="4.3" minGap="0" maxSpeed="27.89" departspeed="0" sigma="0.5" />', file=routes)
            print(' <route id="circle_route" edges="1to2 2to3"/>', file=routes)
            # print(f' <vehicle depart="0" id="{self.front_car_id}" route="circle_route" type="npc_car" color="0,0,1" />', file=routes)
            # print(f' <vehicle depart="0" id="{self.myControlledCarId}" route="circle_route" type="ai_car" color="1,0,0" />', file=routes)
            print(f' <flow id="carflow" type="npc_car" beg="0" end="0" number="{str(self.nbrCars-1)}" from="1to2" to="2to3"/>', file=routes)
            print(f' <vehicle depart="1" id="{self.myControlledCarId}" route="circle_route" type="ai_car" color="1,0,0" />', file=routes)
            print('</routes>', file=routes)

    def get_options(self): #Get option from the xml file that contains the network under the sumo Directory
        opt_parser = optparse.OptionParser()
        opt_parser.add_option("--nogui", action="store_true",
                              default=False, help="run the commandline version of sumo")

        opt_parser.add_option('--log', action="store_true",
                              default=False, help="verbose warning & error log to file")

        opt_parser.add_option('--debug', action="store_true",
                              default=False, help="run the debug mode")

        opt_parser.add_option('--remote', action="store_true",
                              default=False, help="run remote")

        opt, args = opt_parser.parse_args()
        return opt

    def set_parameters(self, options): # Set parameter of the network.sumocfg under the sumo Directory
        if options.nogui:
            sumo_binary = checkBinary('sumo')
        else:
            sumo_binary = checkBinary('sumo-gui')

        params = [sumo_binary, "-c", "sumo/network.sumocfg"]

        if not options.nogui:
            params.append("-S")
            params.append("--gui-settings-file")
            params.append("sumo/gui-settings.cfg")
            

        if options.log:
            params.append("--message-log")
            params.append("log/message_log")

        if options.debug:
            params.append("--save-configuration")
            params.append("debug/debug.sumocfg")

        if options.remote:
            params.append("--remote-port")
            params.append("9999")
            traci.init(9999)

        return params

    def set_reward_after_collision(self, reward_discription): #Set total reward after collision
        #If our agent will do a collision we will take off -10 from total_reward
        if reward_discription == "collision":
            self.totalRewards -= 10
        
        #If our agent will not respect the security distance we will set teh reward to 0
        elif reward_discription == "security_distance":
            self.totalRewards = 0

    def set_reward_security_dist_volicity(self, spaceHeadway, speed, speedLimit): #Set total reward when the security distance is not respected
        if speedLimit and speed > round(self.speed.get('max') + (1 / 3.6) * 5, 2):
            self.totalRewards /= 10
        else:
            distance = np.absolute(np.min([round(spaceHeadway, 0), self.space_headway.get('max')])
                                - self.get_next_vehicle(self.myControlledCarId, self.space_headway.get('max')))
            self.totalRewards += (self.space_headway.get('max') - distance) / self.space_headway.get('max')
           
            #Display distance
            print(f"distance: {distance}")
        self.totalRewards = round(self.totalRewards, 2)
            