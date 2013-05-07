Ruby Restricted Boltzmann Machine
=================================

This is a Ruby port of
[Ed Chen's Python RBM](https://github.com/echen/restricted-boltzmann-machines)
implementation.

I built this library to learn how
[Restricted Boltzmann Machines](http://en.wikipedia.org/wiki/Restricted_Boltzmann_machine)
(RBM) and [Deep Learning](http://en.wikipedia.org/wiki/Deep_learning) work.
No better way to understanding something than implementing it...

It's best to read
[Ed Chen's blog post](http://blog.echen.me/2011/07/18/introduction-to-restricted-boltzmann-machines/)
and follow along with this Ruby code instead of the Python code in his example.

Take it for a test drive
------------------------

Run it like so in a Ruby irb console in the root directory of the repo:

```ruby
require "#{ Dir.pwd }/restricted_boltzmann_machine.rb"
require 'pp'

# Initialize and train the RBM
rbm = RestrictedBoltzmannMachine.new(6, 2)
training_data = [
  [1,1,1,0,0,0],
  [1,0,1,0,0,0],
  [1,1,1,0,0,0],
  [0,0,1,1,1,0],
  [0,0,1,1,1,0],
  [0,0,1,1,1,0]
]
rbm.train(training_data, 10000)

# Inspect the trained weights
# NOTE: The weights here will be different from the ones in Ed's example.
# They could be transposed, however they should converge on the same clusters
# as Ed's.
pp rbm.weights

# Now provide a new input and see what we get on the output.
# You can try a few different inputs that somewhat resemble the movie category
# clusters from Ed's example and you should get consistent output.
user_input = [[0,0,0,1,1,0]]
pp rbm.run_visible(user_input)

# And now let it dream for a bit. It starts with a random input and then
# converges on patterns it's familiar with through training.
pp rbm.daydream(10)
```

About RBMs
----------

What's fascinating about RBMs is that they use two phases of information
processing for learning: a wake phase where they take inputs and translate
them to hidden output states, and a dream phase where they generate visible inputs
based on what they have learned. When you look at the representations of these
inputs, they look a lot like dream images: We can recognize certain patterns,
however they are re-composed in new and unexpected ways.

The other interesting aspect is that an RBM becomes better (meaning more robust)
by adding a probabilistic element to its learning. A similar thing happens in
the human brain where the signals from one Neuron to the next can have a
random delay.
