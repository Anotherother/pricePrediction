{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import numpy\n",
    "\n",
    "# Прибыль моделек\n",
    "\n",
    "def classifyALL(data):\n",
    "    ## 2 - Buy Class, 1 - Sell Class, 0 - Hold Class\n",
    "    \n",
    "    label = []\n",
    "    for i in range(len(data)):     \n",
    "        if i ==0:\n",
    "            label.append(2)\n",
    "        else:\n",
    "            price = data[i:i+2]\n",
    "            label.append(2 * (price[-1] > (price[0])) + 1 * (price[-1] < (price[0])))\n",
    "    label = np.array(label)\n",
    "    label[len(data)-1] = 1\n",
    "    return np.array(label)\n",
    "\n",
    "def calcDOXOD(data, labels):\n",
    "    s = 0\n",
    "    buffer = 0\n",
    "    for i in range (len(data)):\n",
    "        if (i == 0):\n",
    "            s = s - data[i]\n",
    "            \n",
    "        elif (labels[i] == 2 and labels[i-1] != 2 and buffer == 1):\n",
    "            s= s - data[i]\n",
    "            buffer = 0\n",
    "            \n",
    "        elif (labels[i] == 2 and labels[i-1] == 2):\n",
    "            i+=1\n",
    "        \n",
    "        elif (labels[i] == 1 and buffer == 0):\n",
    "            s= s + data[i]\n",
    "            buffer = 1\n",
    "\n",
    "        elif labels[i] == 0:\n",
    "            pass\n",
    "        \n",
    "        if (i == len(data)):\n",
    "            s= s + data[i]\n",
    "\n",
    "    return s"
   ]
  }
 ],
 "metadata": {
  "celltoolbar": "Tags",
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
