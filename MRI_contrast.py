from IPython.display import Image, display
import ipywidgets
import numpy as np
import matplotlib.pyplot as P

seq_info_dict={
    'spin echo': {
        'slider_show': ['TR (s)','TE (s)'],
        'value_init': [.5,.05],
        'slider_range': [[0,10],[0,0.5]],
    },
    'inversion recovery':  {
        'slider_show': ['TR (s)','TE (s)','TI (s)'],
        'value_init': [.5,.5,.5],
        'slider_range': [[0,10],[0,0.5],[0,10]],
    },
    'gradient echo':{
        'slider_show': ['TR (s)','TE (s)'],
        'value_init': [.5,.5],
        'slider_range': [[0,10],[0,0.5]],
    }
}

class MRI_contrast_gui:
    """Class for the GUI for editing automatic segmentation proposals"""
    def __init__(self,seq_name):
        """
        seq_name: name of sequence to work with
        """
        if seq_name not in seq_info_dict:
            print('Sequence not found!')
            return False
        else:
            self.seq_name=seq_name
            self.seq_info=seq_info_dict[self.seq_name]
        self.plt2D=None

        try:
            phantom = np.load('BrainStandardResolution.npz',allow_pickle=True) # location in google colab
        except: 
            phantom = np.load('numPhantom/BrainStandardResolution.npz',allow_pickle=True) # offline location

        ind_show=50
        T1= phantom['T1'][50]
        
        #initialize data
        #Proton density
        self.pd = phantom['Rho'][ind_show].T
        #T1map
        self.T1map = phantom['T1'][ind_show].T
        self.T1map[self.T1map==0]=1
        #T2map
        self.T2map = phantom['T2'][ind_show].T
        self.T2map[self.T2map==0]=1
        #T2starmap
        self.T2starmap = phantom['T2Star'][ind_show].T
        self.T2starmap[self.T2starmap==0]=1

        self.signal=np.copy(self.pd)
        
        # window levels for image contrast
        self.im_min = 0. 
        self.im_max = 1.
        self.figure=P.figure()
        self.sub=self.figure.add_subplot(111)

        self.slider_list=[]
        self.val_list=[]
        
        # MR sequence parameter sliders (add TE, FA etc.)
        for i_st,slider_type in enumerate(self.seq_info['slider_show']): 
            new_slider = ipywidgets.FloatSlider(
                value=self.seq_info['value_init'][i_st],
                min=self.seq_info['slider_range'][i_st][0],
                max=self.seq_info['slider_range'][i_st][1],
                #make 100 steps
                step=self.seq_info['slider_range'][i_st][1]/100,
                description='{}:'.format(slider_type),
                disabled=False,
                continuous_update=True,
                orientation='horizontal',
                readout=True,
                readout_format='.2f',
                layout=ipywidgets.Layout(width='70%'),
            )
            self.slider_list.append(new_slider)
            new_slider.observe(self.change_par, names='value')
            self.val_list.append(new_slider.value)
        display(ipywidgets.VBox(self.slider_list))
        self.updateMRIContrast()

    def change_par(self, w):
        """Callback for the subject selection dropdown. Loads the image and segmentation data for the selected subject."""
        index_owner = self.slider_list.index(w.owner)
        self.val_list[index_owner] = w['new']
        # Update figure
        self.updateMRIContrast()

    # Update image
    def updateMRIContrast(self):
        """Clears the canvas and redraws with the image and segmentation for the slice with index ix in the axes a"""
        #print(self.val_list)
        if self.seq_name == 'spin echo': 
            # spin echo
            TR=self.val_list[0]
            TE=self.val_list[1]
            self.signal = self.signal_spin_echo(self.pd,self.T1map,self.T2map,
                                                TR,TE)
        elif self.seq_name == 'inversion recovery':
            TR=self.val_list[0]
            TE=self.val_list[1]
            TI=self.val_list[2]
            # inversion recovery
            self.signal = self.signal_inversion_recovery(
                self.pd,self.T1map,self.T2map,
                TR,TE,TI)
        elif self.seq_name == 'gradient echo':
            TR=self.val_list[0]
            TE=self.val_list[1]
            # inversion recovery
            self.signal = self.signal_gradient_echo(
                self.pd,self.T1map,self.T2starmap,TR,TE)

        # Show updated image contrast
        if self.plt2D:
            self.plt2D.set_data(self.signal)
            self.sub.draw_artist(self.plt2D)
        else:
            self.plt2D= self.sub.imshow(self.signal,cmap='gray',vmin=self.im_min, vmax=self.im_max)
            self.sub.set_axis_off()

    def signal_spin_echo(self,pd,T1,T2,TR,TE):
        """
        pd: proton density []
        T1: T1 relaxation time [ms]
        T2: T2 relaxation time [ms]
        TR: Repetition time [ms]
        TE: echo time [ms]
        """
        return np.abs(pd*np.exp(-TE/T2)*(1-2*np.exp(-TR/T1)))

    def signal_gradient_echo(self,pd,T1,T2star,TR,TE):
        """
        pd:     proton density []
        T1:     T1 relaxation time [ms]
        T2star: T2* relaxation time [ms]
        TR:     Repetition time [ms]
        TE:     echo time [ms]
        """
        return np.abs(pd*np.exp(-TE/T2star)*(1-2*np.exp(-TR/T1)))

    def signal_inversion_recovery(self,pd,T1,T2star,TR,TE,TI):
        """
        pd:     proton density []
        T1:     T1 relaxation time [ms]
        T2star: T2* relaxation time [ms]
        TI:     inversion time [ms]
        TR:     Repetition time [ms]
        TE:     echo time [ms]
        """
        return np.abs(pd*np.exp(-TE/T2star)*(1-2*np.exp(-TR/T1))*
                      (1-2*np.exp(-TR/TI)))

# debug
#if True:
if False:
    gui1 = MRI_contrast_gui('spin echo')
    gui2 = MRI_contrast_gui('inversion recovery')
    gui3 = MRI_contrast_gui('gradient echo')
    #gui.changeTR({'new': 5.0})
