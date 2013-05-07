require_relative 'random_gaussian.rb'
require 'matrix'

class RestrictedBoltzmannMachine

  attr_accessor :num_hidden
  attr_accessor :num_visible
  attr_accessor :learning_rate
  attr_accessor :weights

  # Initializes a new RBM instance.
  # @param[Integer] num_visible Number of visible units
  # @param[Integer] num_hidden Number of hidden units
  # @param[Float, optional] learning_rate Learning rate, defaults to 0.1
  def initialize(num_visible, num_hidden, learning_rate = 0.1)
    @num_hidden = num_hidden
    @num_visible = num_visible
    @learning_rate = learning_rate

    gauss_rng = RandomGaussian.new(0.0, 0.1)
    # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
    # a Gaussian distribution with mean 0 and standard deviation 0.1.
    @weights = Array.new(num_visible) { Array.new(num_hidden) { gauss_rng.rand } }
    # Insert weights for the bias units into the first row and first column.
    @weights.each { |row| row.unshift(0) }
    @weights.unshift(Array.new(num_hidden + 1, 0))
  end

  # Train the RBM
  # @param[Array<Array<1,0>>] training_samples Training samples: A matrix where each
  #                           row is a training sample with 1s and 0s, one for
  #                           each visible unit.
  # @param[Integer, optional] max_epochs Number of training epochs
  def train(training_samples, max_epochs = 1000)
    training_samples = Marshal.load(Marshal.dump(training_samples)) # can't use dup since it's just a shallow copy
    num_examples = training_samples.length

    # Insert bias units of 1 into the first column of each training_sample.
    training_samples.each { |training_sample| training_sample.unshift(1) }

    training_samples_matrix = Matrix[*training_samples]
    max_epochs.times do |epoch|
      # Clamp to the training_samples and sample from the hidden units.
      # (This is the "positive CD phase", aka the reality phase.)
      pos_hidden_activations = training_samples_matrix * Matrix[*@weights]
      pos_hidden_probs = pos_hidden_activations.map { |e| logistic_function(e) }
      pos_hidden_states = pos_hidden_probs.map { |e| e > rand ? 1.0 : 0.0 }
      # Note that we're using the activation *probabilities* of the hidden states, not the hidden states
      # themselves, when computing associations. We could also use the states; see section 3 of Hinton's
      # "A Practical Guide to Training Restricted Boltzmann Machines" for more.
      pos_associations = training_samples_matrix.transpose * Matrix[*pos_hidden_probs]

      # Reconstruct the visible units and sample again from the hidden units.
      # (This is the "negative CD phase", aka the daydreaming phase.)
      neg_visible_activations = Matrix[*pos_hidden_states] * Matrix[*@weights].transpose
      neg_visible_probs = neg_visible_activations.map { |e| logistic_function(e) }

      neg_visible_probs = Matrix[*neg_visible_probs.to_a.map { |row| row[0] = 1; row }] # Fix the bias unit to 1
      neg_hidden_activations = neg_visible_probs * Matrix[*@weights]
      neg_hidden_probs = neg_hidden_activations.map { |e| logistic_function(e) }
      # Note, again, that we're using the activation *probabilities* when computing associations, not the states
      # themselves.
      neg_associations = neg_visible_probs.transpose * neg_hidden_probs

      # Update weights.
      @weights = Matrix[*@weights] + (
        @learning_rate * ((pos_associations - neg_associations) / num_examples)
      )

      error = ((training_samples_matrix - neg_visible_probs).map { |e| e ** 2 }).reduce(:+)
      puts "Epoch #{ epoch }: error is #{ error }"
    end
  end

  # Assuming the RBM has been trained (so that weights for the network have
  # been learned), run the network on a set of visible units, to get a sample
  # of the hidden units.
  # @param[Array<Array<0,1>>] data A matrix where each row consists of the
  #                           states of the visible units.
  # @return[Array<Array<>>] hidden_states A matrix where each row consists of
  #                         the hidden units activated from the visible units
  #                         in the data matrix passed in.
  def run_visible(data)
    num_examples = data.length

    # Create a matrix, where each row is to be the hidden units (plus a bias unit)
    # sampled from a training example.
    hidden_states = Matrix.build(num_examples, (num_hidden + 1)){ 1 }


    # Insert bias units of 1 into the first column of data.
    data.each { |row| row.unshift(1) }

    # Calculate the activations of the hidden units.
    hidden_activations = Matrix[*data] * Matrix[*@weights]
    # Calculate the probabilities of turning the hidden units on.
    hidden_probs = hidden_activations.map { |e| logistic_function(e) }
    # Turn the hidden units on with their specified probabilities.
    hidden_states = hidden_probs.map { |e| e > rand ? 1.0 : 0.0 }

    # Ignore the bias units.
    hidden_states.to_a.map { |row| row[1..-1] }
  end

  # Assuming the RBM has been trained (so that weights for the network have been learned),
  # run the network on a set of hidden units, to get a sample of the visible units.
  # @param[Array<Array<>>] data A matrix where each row consists of the states of the hidden units.
  # @param[Array<Array<>>] visible_states A matrix where each row consists of
  #                        the visible units activated from the hidden units in
  #                        the data matrix passed in.
  # TODO: Remove the code duplication between this method and `run_visible`?
  def run_hidden(data)
    num_examples = data.length

    # Create a matrix, where each row is to be the visible units (plus a bias unit)
    # sampled from a training example.
    visible_states = Matrix.build(num_examples, (num_visible + 1)){ 1 }

    # Insert bias units of 1 into the first column of data.
    data.each { |row| row.unshift(1) }

    # Calculate the activations of the visible units.
    visible_activations = Matrix[*data] * Matrix[*@weights].transpose

    # Calculate the probabilities of turning the visible units on.
    visible_probs = visible_activations.map { |e| logistic_function(e) }
    # Turn the visible units on with their specified probabilities.
    visible_states = visible_probs.map { |e| e > rand ? 1.0 : 0.0 }

    # Ignore the bias units.
    visible_states.to_a.map { |row| row[1..-1] }
  end

  # Randomly initialize the visible units once, and start running alternating
  # Gibbs sampling steps (where each step consists of updating all the hidden
  # units, and then updating all of the visible units), taking a sample of the
  # visible units at each step.
  # Note that we only initialize the network *once*, so these samples are correlated.
  # @param[Integer] num_samples The number of wake/dream cycles to run
  # @return [Array<Array>] samples: A matrix, where each row is a sample of the
  #                        visible units produced while the network was
  #                        daydreaming.
  def daydream(num_samples)

    # Create a matrix, where each row is to be a sample of of the visible units
    # (with an extra bias unit), initialized to all ones.
    samples = num_samples.times.map { (num_visible + 1).times.map { 1 } }

    # Take the first sample from a uniform distribution.
    samples[0] = (num_visible + 1).times.map { rand }

    # Start the alternating Gibbs sampling.
    # Note that we keep the hidden units binary states, but leave the
    # visible units as real probabilities. See section 3 of Hinton's
    # "A Practical Guide to Training Restricted Boltzmann Machines"
    # for more on why.
    num_samples.times do |i|
      visible = samples[i]
      # Calculate the probabilities of turning the hidden units on.
      hidden_activations = Matrix[visible] * Matrix[*@weights]
      hidden_probs = hidden_activations.map { |e| logistic_function(e) }
      hidden_states = hidden_probs.map { |e| e > rand ? 1.0 : 0.0 }
      # Always fix the bias unit to 1.
      hidden_states = hidden_states.to_a
      hidden_states.each { |row| row[0] = 1 }
      # Recalculate the probabilities that the visible units are on.
      visible_activations = Matrix[*hidden_states] * Matrix[*@weights].transpose
      visible_probs = visible_activations.map { |e| logistic_function(e) }
      visible_states = visible_probs.map { |e| e > rand ? 1.0 : 0.0 }
      # record state of visible unites in next sample
      samples[i+1] = visible_states.to_a.first
    end

    # Ignore the bias units.
    samples.to_a.map { |row| row[1..-1] }
  end

  def logistic_function(x)
    1.0 / (1.0 + Math.exp(-x))
  end

end
