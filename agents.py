import numpy as np
import mesa
import random

class Household(mesa.Agent):
    """Household agents"""

    def __init__(self, model):
        # pass the params to parent class
        super().__init__(model)

        self.w_h = 1.0 # reservation wage
        self.m_h = 0.0 # liquidity
        self.type_a_connections = [] # list of firms 
        self.type_b_connection = None # employment
        self.c_r_h = 0 # demand
        self.income = 0 #income from last month

    def monthly_consumption(self, alpha):
        """
        find each households monthly consumption need. 
        - find avg of all trade connection prices
        - use eq 12 to find consumption
        """
        prices = 0
        for a in self.type_a_connections:
            prices += a.p_f
        p_i_h = prices / len(self.type_a_connections)
        self.c_r_h = min((self.m_h / p_i_h)**alpha, self.m_h / p_i_h)
    
    def buy_goods(self, n):
        """
        each household buys goods from a random type a connection
        - if they cant satisfy the demand - move on to another random type a connection
        - if hh cant afford - buy all they can afford 
        - stop when demand 95% satisfied
        """
        demand = self.c_r_h / 21
        og_demand = demand
        shops = random.sample(self.type_a_connections, len(self.type_a_connections))
        for shop in shops[:n]:
            if shop.i_f >= demand and self.m_h >= (shop.p_f * demand):
                self.m_h -= shop.p_f * demand
                shop.m_f += shop.p_f * demand
                shop.i_f -= demand
                demand = 0
            elif self.m_h < (shop.p_f * demand):
                demand_new = self.m_h / shop.p_f
                self.m_h -= shop.p_f * demand_new
                shop.m_f += shop.p_f * demand_new
                shop.i_f -= demand_new
                demand -= demand_new
            elif shop.i_f < demand:
                self.m_h -= shop.p_f * shop.i_f
                shop.m_f += shop.p_f * shop.i_f
                demand -= shop.i_f
                shop.i_f = 0
            if demand <= 0.05 * og_demand:
                break   

    def adjust_reservation_wage(self):
        """ adjust reservation wage according to last months income"""
        if self.income > self.w_h:
            self.w_h = self.income
        if self.income == 0:
            self.w_h = self.w_h * 0.9

class Firm(mesa.Agent):
    """Firm agents"""

    def __init__(self, model):
        super().__init__(model)

        self.m_f = 0.0 # liquidity
        self.i_f = 0.0 # inventory
        self.p_f = 1.0 # goods price
        self.w_f = 1.0 # wage rate
        self.workers = [] # list of workers
        self.l_f = len(self.workers) # no of workers?
        self.buffer = 0
        self.n_positions = 10
        self.months_full = 0

    def produce(self, ld):
        """produce goods"""
        self.l_f += ld * self.l_f

    def pay_wages(self):
        """
        firm pays wages to all workers
        either at set wage rate or if cant afford, drop wage
        """
        # add buffer to liquidity and then reset to 0
        self.m_f += self.buffer
        self.buffer = 0
        if self.m_f >= self.w_f * self.l_f:
            for h in self.workers:
                self.m_f -= self.w_f
                h.m_h += self.w_f
                # store income
                h.income = self.w_f
        else:
            new_wage = self.m_f / self.l_f
            for h in self.workers:
                self.m_f -= new_wage
                h.m_h += new_wage
                # store income
                h.income = new_wage

    def add_to_buffer(self, chi):
        """add left over liquidity to buffer """
        if self.m_f > chi * self.w_f * self.l_f:
            # add to buffer - full amt
            self.buffer += chi * self.w_f * self.l_f
            self.m_f -= chi * self.w_f * self.l_f
        elif self.m_f > 0:
            # add to buffer - not full amt
            self.buffer += self.m_f
            self.m_f = 0

    def distribute_profits(self):
        """distribute remaining profit to all hh according to their liquidity"""
        if self.m_f > 0: 
            # distribute profit for each house proportional to m_h
            households = self.model.agents.select(agent_type=Household)
            total_liquid = sum(h.m_h for h in households)
            for h in households:
                h.m_h += (self.m_f / total_liquid) * h.m_h
        self.m_f = 0

    def set_wages(self, gamma, delta):
        """
        set wages for the month
        """
        # draw from uniform dist
        mu = random.uniform(0, delta)
        if self.n_positions > self.l_f:
            self.w_f = self.w_f * (1 + mu)
            self.months_full = 0

        elif self.months_full > gamma:
            self.w_f = self.w_f * (1 - mu)
            self.months_full += 1
        else:
            self.months_full += 1
        

        




