Setup Reference
===============

Typical simulator settings
--------------------------

.. list-table::
   :header-rows: 1

   * - Parameter
     - Meaning
   * - ``T``
     - simulation length
   * - ``Ts``
     - control-loop sampling period
   * - ``fs_vib``
     - vibration sampling frequency
   * - ``J``
     - inertia
   * - ``B``
     - damping
   * - ``K``
     - stiffness
   * - ``Kp``
     - proportional gain
   * - ``Ki``
     - integral gain
   * - ``Kd``
     - derivative gain
   * - ``mag_levels``
     - fault magnitude levels
   * - ``N_TRIALS``
     - repetitions per regime
   * - ``BASE_SEED``
     - random seed base

LNU/PCA settings
----------------

.. list-table::
   :header-rows: 1

   * - Parameter
     - Meaning
   * - ``nd``
     - number of reference delays
   * - ``nu``
     - number of control delays
   * - ``ny``
     - number of output delays
   * - ``L_u``
     - control delay
   * - ``NPCA``
     - number of samples used for PCA
   * - ``PCA_VAR_THRESHOLD``
     - retained PCA variance
   * - ``MU``
     - LM regularization/scaling parameter
   * - ``EPOCHS``
     - number of training iterations

MLP settings
------------

.. list-table::
   :header-rows: 1

   * - Parameter
     - Meaning
   * - ``HIDDEN_LAYER_SIZES``
     - hidden layer structure
   * - ``LEARNING_RATE``
     - optimizer learning rate
   * - ``N_EPOCHS``
     - number of training epochs
   * - ``TEST_SIZE``
     - validation/test split
   * - ``RANDOM_SEED``
     - reproducibility seed
