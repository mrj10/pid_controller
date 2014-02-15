#!/usr/bin/env python
import sys
from collections import namedtuple, deque
import heapq
import numpy as np
import matplotlib.pyplot as plt

class PIDSimEventBase:
	def __init__(self):
		pass
	def apply(self, eventq, current_timestep, all_objects, connections):
		pass
	def __lt__(self, other):
         return self.__class__.order_key < other.__class__.order_key
	order_key = 0;

class PIDSimEventPutIn(PIDSimEventBase):
	def __init__(self, object):
		self.object = object
	def apply(self, eventq, current_timestep, all_objects, connections):
		all_objects.append(self.object)
	order_key = 1

class PIDSimEventTakeOut(PIDSimEventBase):
	def __init__(self, object):
		self.object = object
	def apply(self, eventq, current_timestep, all_objects, connections):
		all_objects.remove(self.object)
		#Remove all connections this object was a part of
		#FIXME faster way to search for self.object?
		for pair in connections:
			if pair[0] == self.object or pair[1] == self.object:
				del connections[pair]
	order_key = 1

class PIDSimEventPerturbEnergy(PIDSimEventBase):
	def __init__(self, object, energy_change):
		self.object = object
		self.energy_change = energy_change
	def apply(self, eventq, current_timestep, all_objects, connections):
		if self.object in all_objects:
			self.object.temp = self.object.temp + (self.energy_change / (self.object.mass * self.object.specific_heat))
	order_key = 2

class PIDSimEventPerturbTemperature(PIDSimEventBase):
	def __init__(self, object, temp_change):
		self.object = object
		self.temp_change = temp_change
	def apply(self, eventq, current_timestep, all_objects, connections):
		if self.object in all_objects:
			self.object.temp = self.object.temp - self.temp_change
	order_key = 2

class PIDSimEventPerturbMass(PIDSimEventBase):
	def __init__(self, object, mass_change, absolute=True):
		self.object = object
		self.mass_change = mass_change
		self.absolute = absolute
	def apply(self, eventq, current_timestep, all_objects, connections):
		if self.object in all_objects:
			if self.absolute:
				self.object.mass = self.object.mass + self.mass_change
			else:
				self.object.mass = self.object.mass * self.mass_change
	order_key = 2

class PIDSimEventAddConnection(PIDSimEventBase):
	def __init__(self, obj1, obj2, seconds_to_99_percent_transfer):
		self.obj1 = obj1
		self.obj2 = obj2
		#If e^(-r*t) = .99, r=.01005/t
		self.r = .01005/seconds_to_99_percent_transfer
	def apply(self, eventq, current_timestep, all_objects, connections):
		if self.obj1 in all_objects and self.obj2 in all_objects:
			connections[self.obj1, self.obj2] = self.r
	order_key = 2

class PIDSimObject:
	def __init__(self, name, mass, specific_heat, temp):
		self.name = name
		self.mass = mass
		self.specific_heat = specific_heat
		self.temp = temp

class PIDSimEventInstance:
	def __init__(self, event, timestep):
		self.event = event
		self.timestep = timestep
	def __lt__(self, other):
		if self.timestep != other.timestep:
			return self.timestep < other.timestep
		else:
			return self.event < other.event

class PIDSimObjectGraph(dict):
	def __getitem__(self, key, second=None):
		return super(PIDSimObjectGraph, self).__getitem__(tuple(sorted(key)))

	def __setitem__(self, key, value):
		super(PIDSimObjectGraph, self).__setitem__(tuple(sorted(key)), value)

def print_all_objects(ts, objects):
	print 'Timestep %d' % (ts)
	for o in objects:
		print '%s, temp %f C, mass %fg, specific heat %fg' % (o.name, o.temp, o.mass, o.specific_heat)

def add_connection(connections, obj1, obj2, seconds_to_99_percent_transfer):
	#If e^(-r*t) = .99, r=.01005/t
	connections[obj1, obj2] = .01005/seconds_to_99_percent_transfer

def sim(timestep_size, num_timesteps, events):
	eventq = [] #Managed with heappush() and heappop() from heapq
	objects = []
	connections = PIDSimObjectGraph()
	temps = {}
	controller = PIDController((.01, -10, 10), (.001, -10, 10), (.005, -10, 10), 0, 1)

	for event in events:
		heapq.heappush(eventq, event)
	for t in xrange(num_timesteps):
		#Apply all events for this timestep
		while eventq and eventq[0].timestep == t:
			eventinstance = heapq.heappop(eventq)
			eventinstance.event.apply(eventq, t, objects, connections)
		#Heat diffusion
		#For each pair of connected objects
		for pair in connections:
			#print 'Processing ', pair[0].name, '/', pair[1].name, ' connection @ ts ', t
			if pair[0].temp > pair[1].temp:
				hotter, colder = pair[0], pair[1]
			else:
				hotter, colder = pair[1], pair[0]
			h = connections[pair] #Value in dict if heat transfer coefficient
			dQdt = h * (hotter.temp - colder.temp)
			dQ = dQdt * timestep_size
			hotter.temp = hotter.temp - (dQ / (hotter.mass*hotter.specific_heat))
			colder.temp = colder.temp + (dQ / (colder.mass*colder.specific_heat))
		#PID control comes last
		water = find(lambda obj: obj.name == 'Water', objects)
		output = controller.control(65, water.temp, (t%100 == 0))
		heater_wattage = 300
		PIDSimEventPerturbEnergy(water, heater_wattage*timestep_size*output).apply(events, t, objects, connections)
		temps.setdefault('pid', []).append((t, output))

		if t % 100 == 0:
			print_all_objects(t, objects)
		for o in objects:
			temps.setdefault(o, []).append((t, o.temp),)
	return temps

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

def find(f, seq):
  """Return first item in sequence where f(item) == True."""
  for item in seq:
    if f(item): 
      return item

def round_partial(value, resolution):
    return round(value / resolution) * resolution

class PIDController:
	qlen = 5
	#x_params are each 3-tuples of (scale factor, min val, max val)
	def __init__(self, p_params, i_params, d_params, min, max):
		#Circular buffer of last N error values (for filtering and derivatives)
		self.errorq = deque(maxlen=PIDController.qlen)
		for i in range(PIDController.qlen-1):
			self.errorq.appendleft(0)

		self.p_scale = p_params[0]
		self.p_min = p_params[1]
		self.p_max = p_params[2]
		self.i_scale = i_params[0]
		self.i_min = i_params[1]
		self.i_max = i_params[2]
		self.d_scale = d_params[0]
		self.d_min = d_params[1]
		self.d_max = d_params[2]

		self.min = min
		self.max = max

		self.i = 0

		#Circular buffer of last N output values (for e.g. filtering)
		self.outputq = deque(maxlen=PIDController.qlen)
		for i in range(PIDController.qlen-1):
			self.outputq.appendleft(0)
	#Returns an output value between 0 and 1
	def control(self, set_value, present_value, verbose=False):
		#Assume we can only read present_value to within 1 14-bit LSB with full range of 300C
		present_value_rounded = round_partial(present_value, 300.0/(2**14))
		error = set_value - present_value_rounded
		self.errorq.appendleft(error)

		p = clamp(self.p_scale * error, self.p_min, self.p_max)
		
		self.i = clamp(self.i + error, self.i_min, self.i_max)
		
		d = clamp(self.d_scale * ((3*self.errorq[0] - 4*self.errorq[1] + 1*self.errorq[2]) / 2.0), self.d_min, self.d_max)

		output = clamp(p + self.i + d, self.min, self.max)
		self.outputq.appendleft(output)

		if verbose:
			print 'SV=%f PV=%f error=%f p=%f i=%f d=%f out=%f' % (set_value, present_value, error, p, self.i, d, output)
		return output

def main(argv=None):
	if argv is None:
		argv = sys.argv
	pottemp = 30
	#12 quarts of water
	water = PIDSimObject('Water', 10886.2, 4.186, pottemp)
	#2 pounds of pork loin (see http://www.engineeringtoolbox.com/specific-heat-capacity-food-d_295.html)
	pork = PIDSimObject('Pork', 907.185, 2.76, 5)
	#2kg of stainless steel
	pot = PIDSimObject('Pot', 2000.0, 0.51, pottemp)
	#Environment around pot, assume 65f and infinite mass (will never change in temperature)
	air = PIDSimObject('Air', float('inf'), 1.006, 18.33)
	events = []
	events.append(PIDSimEventInstance(PIDSimEventPutIn(water), 0))
	events.append(PIDSimEventInstance(PIDSimEventPutIn(pork), 0))
	events.append(PIDSimEventInstance(PIDSimEventPutIn(pot), 0))
	events.append(PIDSimEventInstance(PIDSimEventPutIn(air), 0))
	events.append(PIDSimEventInstance(PIDSimEventAddConnection(water, pork, .01), 0))
	events.append(PIDSimEventInstance(PIDSimEventAddConnection(water, pot, .01), 0))
	events.append(PIDSimEventInstance(PIDSimEventAddConnection(pot, air, .25), 0))
	events.append(PIDSimEventInstance(PIDSimEventAddConnection(water, air, .08), 0))

	results = sim(1, 5000, events)
	for o, l in results.iteritems():
		times, temps = zip(*l)
		#Keys are either strings or PIDSimObjects
		#Strings are used for other miscellaneous quantities we want to track and plot
		line, = plt.plot(times, temps, '-', linewidth=2, label=(o.name if isinstance(o, PIDSimObject) else o))
	plt.legend(loc=4) #Lower right (see http://matplotlib.org/users/legend_guide.html)
	plt.show()


if __name__ == "__main__":
    sys.exit(main())