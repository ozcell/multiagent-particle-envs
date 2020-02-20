import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

import pdb


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.state.false_l_pos = []
            for i, landmark in enumerate(world.landmarks):
                agent.state.false_l_pos.append(np.random.uniform(-1, +1, world.dim_p))
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        # dists = []
        # for i_agent in world.agents:
        #     dists.append(np.square(i_agent.state.p_pos).sum())
        # leader = np.argsort(dists)
        # world.leader = leader

        #leader = []
        #if np.random.randint(2) == 0:
        #    leader.append([0,1,2])
        #else:
        #    leader.append([2,0,1])

        leader = np.arange(3)
        np.random.shuffle(leader)

        #leader.append(np.random.randint(1,3))
        #if leader[0] == 1:
        #    if np.random.randint(2) == 0:
        #        leader.append(0)
        #        leader.append(2)
        #    else:
        #        leader.append(2)
        #        leader.append(0)
        #elif leader[0] == 2:
        #    leader.append(0)
        #    leader.append(1)
        world.leader = leader
        #world.agents[leader].color = np.array([0.85, 0.35, 0.35])

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in world.landmarks:
            # if l is world.landmarks[0]:
            #     dist = np.sqrt(np.sum(np.square(world.agents[(world.leader[0]+1)%len(world.agents)].state.p_pos - l.state.p_pos)))
            # elif l is world.landmarks[1]:
            #     dist = np.sqrt(np.sum(np.square(world.agents[(world.leader[1]+1)%len(world.agents)].state.p_pos - l.state.p_pos)))
            # elif l is world.landmarks[2]:
            #     dist = np.sqrt(np.sum(np.square(world.agents[(world.leader[2]+1)%len(world.agents)].state.p_pos - l.state.p_pos)))
            if l is world.landmarks[0]:
                dist = np.sqrt(np.sum(np.square(world.agents[1].state.p_pos - l.state.p_pos)))
            elif l is world.landmarks[1]:
                dist = np.sqrt(np.sum(np.square(world.agents[2].state.p_pos - l.state.p_pos)))
            elif l is world.landmarks[2]:
                dist = np.sqrt(np.sum(np.square(world.agents[0].state.p_pos - l.state.p_pos)))
            rew -= dist
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        flag = []
        if agent is world.agents[world.leader[0]]:
            flag.append(np.ones(1)*1)
            for i_entity, entity in enumerate(world.landmarks):
                if entity is world.landmarks[0]:
                    entity_pos.append(entity.state.p_pos - agent.state.p_pos)
                else:
                    entity_pos.append(agent.state.false_l_pos[i_entity] - agent.state.p_pos)
        elif agent is world.agents[world.leader[1]]:
            flag.append(np.ones(1)*2)
            for i_entity, entity in enumerate(world.landmarks):
                if entity is world.landmarks[1]:
                    entity_pos.append(entity.state.p_pos - agent.state.p_pos)
                else:
                    entity_pos.append(agent.state.false_l_pos[i_entity] - agent.state.p_pos)
        elif agent is world.agents[world.leader[2]]:
            flag.append(np.ones(1)*3)
            for i_entity, entity in enumerate(world.landmarks):
                if entity is world.landmarks[2]:
                    entity_pos.append(entity.state.p_pos - agent.state.p_pos)
                else:
                    entity_pos.append(agent.state.false_l_pos[i_entity] - agent.state.p_pos)

        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + flag)
		
