#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
31/01/2025: * Bias-T eklendi. 

08/07/2024: * Sadece thread olarak çalışan bu programda sdr_parameters ve demod_parameters
              olarak adlandırdığım değişkene ayrıca gerek olmadığı için o değişkenleri
              program mantığından çıkardım, ancak hatırlatmak amacı ile comment şeklinde tutuyorum.
            * Birkaç yerde yazım hataları ve formatta düzeltmeler yaptım.
            
26/04/2024: * Küçük önemsiz değişiklikler yaptım.

21/04/2024: * Sampling_rate parametresine göre rf nr pencere boyunun ayarlanması sağlandı
              ve bir kaç comment edilmiş yer yeniden aktif hale getirildi.
              
02/04/2024: * multiprocessing queue yerine normal queue kullanıldı. Multiprocessing
              queue prosesler arasında haberleşme için var olması sebebiyle, 
              içine veri koyulmadan veya veri alınmadan önce serileştirildiği için
              (pickling/unpickling) normalinden daha yavaş çalışmakta. Her ne kadar
              burada etkisini göremesem de daha hızlı olan normal versiyonunu
              tüm program aynı bellek uzayında olduğu için burada tercih ettim.
              
24/03/2024: * Her konvolüsyon işlemi için ilkleştirme değerlerinin yeniden
              atanması, hemen işlemin ardından olacak şekilde yapıldı. Böylece
              işlem blokları üzerindeki ekleme/çıkarma işi daha kolay yapılır
              hale geldi. 
            * Demodülasyon işlemine ait değişkenler kendi altlarında guruplandı.
            * RF NR pencere boyları örnekleme hızlarına adapte edildi.
              
23/03/2024: * Ufak düzenlemeler yaptım..

22/03/2024: * FM RF Noise Reduction eklendi.

21/03/2024: * AM ve FM demodülatörde açıklamalar eklendi.

20/03/2024: * Font dosyaları çalışılan klasöre taşındı.

06/03/2024: * Tüm değişkenler sınıfa ait globaller olduğu için, demodülatörün
              geri dönüş değerlerinde (derlenmiş fonksiyonlar hariç)
              sadeleştirmeler yaptım. 
              
04/03/2024: * Konvolüsyon ile çalışan radyo programının tek dosya halinde 
              düzenlenmiş olanı. Prosesler thread haline getirildi.
            * Evde düzenlemeler yaptım. Volume çalışıyor artık.
            
05/03/2024: * "spectrum_and_waterfall" processor içine katıldı.
            * Streaming içinde, normalde var olmayan bir parametre düzeltildi.
            * Gereksiz self.num_rows değişkeni kaldırıldı ve self.waterfall dizisi
              doğru parametre olan self.wf_len ile düzeltildi. Artık waterfall 
              grafikleri 65536 olan limit veri değerinden etkilenmiyor. Bunun 
              sebebi sanırım baştan yanlış init edilmesiydi.
            * Bu arada multiprocess olan Radio.py daha hızlı çalışıyormuş. 
              
@author: tansu
"""
import dearpygui.dearpygui as dpg
import multiprocessing as mp
import queue as q
import threading as th
import numpy as np
from collections import deque
from rtlsdr import RtlSdr
import os
import argparse
import sounddevice as sd   
import scipy.signal as s
# import time as t
import hpf_ran, agc_ran, moving_average_ran, spectrum_partial_fft_ran, calculate_vumeter_data_ran
# from Queue_with_shared_memory import SharedMemoryQueue as sq
parser = argparse.ArgumentParser(description='Simple radio program using RTL-SDR and DearPyGUI')
parser.add_argument('--freq', dest='freq', type=float, default=100000000, help='RF Tune Frequency')
args = parser.parse_args()

# ##### MAIN PROGRAM ##### #
class gui():
    def __init__(self):
        self.freq = args.freq  # Tune frekansı.
        self.srate = 960000  # Örnekleme hızı.
        self.arate = 48000  # Ses kartı örnekleme hızı.
        self.data_size = 51200  # SDR'den tek seferde alınan veri miktarı.
        self.window_size = 2048  # Konvolüsyon "kernel" boyu.
        self.nrw_size = 11  # Audio Noise reduction konvolüsyon kernel boyu.
        self.fm_rfnr_size = 12 if self.srate == 960000 else 23
        self.am_rfnr_size = 131 if self.srate == 960000 else 225
        self.rf_nr_state = False
        self.windowed = True  # Konvolüsyon kernellerinin pencere fonksiyonundan geçsin mi, geçmesin mi. Geçmeli.
        self.chunk_size = int(np.rint(self.arate*(self.data_size/self.srate))) # Ses kartına tek seferde giden veri miktarı.
        self.decimation_factor = int(np.rint(self.srate/self.arate))
        self.slevel = 0.0  # Bu artık squelch level
        self.alevel = 0.0  # Audio denoiser level (nrw_size ile artık aynı)
        self.demod_type = None
        self.bw = 16000  # Kanal band genişliği.
        self.f_low = 0  # Kanal bandı alt kesim frekansı
        self.f_high = self.bw # Kanal bandı üst kesim frekansı
        self.volume = 10.0 # Ses kartına gitmeden önce yapılan yükseltme katsayısı.
        self.gain_values = {"1":0.0, "2":0.9, "3":1.4, "4":2.7, "5":3.7, "6":7.7, 
                            "7":8.7, "8":12.5, "9":14.4, "10":15.7, "11":16.6, 
                            "12":19.7, "13":20.7, "14":22.9, "15":25.4, "16":28.0,
                            "17":29.7, "18":32.8, "19":33.8, "20":36.4,
                            "21":37.2, "22":38.6, "23":40.2, "24":42.1, "25":43.4,
                            "26":43.9, "27":44.5, "28":48.0, "29":49.6}
        self.srates = {"960k": 960000, "1920k": 1920000}
        self.window_sizes = {"1024": 1024, "2048": 2048, "FULL": self.data_size}
        self.rf_gain = self.gain_values["12"]
        self.wf_len = 128  # Waterfall grafiği yatay satır sayısı. 
        self.fft_size = 1024  # Spektrum göstermek için alınan fft örnek sayısı.
        self.f_span_min = self.freq-self.srate/2
        self.f_span_max = self.freq+self.srate/2
        self.f_step = self.srate/self.fft_size
        self.spectrum = np.ones(self.fft_size)*(-50)
        self.spec_axis = np.arange(self.f_span_min, self.f_span_max, self.f_step)
        self.wf_bottom = -200*np.ones(self.fft_size)
        self.waterfall = np.zeros((self.wf_len, self.fft_size))
        self.waterfall_array = deque(maxlen=self.wf_len) 
        # SDR Setup..
        self.sdr = RtlSdr()
        self.rf_manual_gain_state = True
        self.sdr.set_manual_gain_enabled(self.rf_manual_gain_state) 
        self.sdr.set_gain(self.rf_gain)
        self.agc_state = 1
        self.sdr.set_agc_mode(self.agc_state)
        self.sdr.set_sample_rate(self.srate)
        self.sdr.set_direct_sampling(0 if self.freq>28800000 else 2)
        self.sdr.set_center_freq(self.freq)
        self.sdr.set_bias_tee(0)
        self.bias_tee_state = 0
               
#         self.sdr_parameters = {"freq": self.freq, "rf_manual_gain_state": self.rf_manual_gain_state, "rf_gain": self.rf_gain, "sdr_agc": self.agc_state, "srate": self.srate, "status": None}
#         self.demod_parameters = {"f_low": self.f_low, "f_high": self.f_high, "srate": self.srate, "dsize": self.data_size, "nrw_size": self.nrw_size,
#                                  "chunk_size": self.chunk_size, "window_size": self.window_size, "slevel": self.slevel, "alevel": self.alevel, 
#                                  "demod_type": self.demod_type, "decimation_factor": self.decimation_factor, "windowed": self.windowed, "rf_nr": self.rf_nr_state, "status": None}
                       
        self.qsdr = q.Queue(2)       
        self.qmes = q.Queue(1)
        self.qspwf = q.Queue(2)
        self.qpro = q.Queue(2)
        self.qpro_mes = q.Queue(1)
        self.qdemod = q.Queue(2)
        self.qdemod_mes = q.Queue(1)
        self.qsnd = q.Queue(2)
        self.qsnd_mes = q.Queue(1)
        self.qvuin = q.Queue(2)
        self.qvuout = q.Queue(2)
        self.qvu_mes = q.Queue(1)
                    
        self.event = mp.Event()
        self.lock = th.Lock()
        
#         self.qdemod_mes.put(self.demod_parameters) # Burası demodülatörün ilkleştirilmesi için gerekli şimdilik..
        self.qdemod_mes.put(1)
        self.th_streamer = th.Thread(target=self.Streaming, args=()) 
        self.th_processor = th.Thread(target=self.processor, args=())
        self.th_updater = th.Thread(target=self.updater, args=())        
        self.th_demodulator = th.Thread(target=self.demodulator, args=())
        self.th_sound = th.Thread(target=self.sound, args=())
        self.th_vumeter = th.Thread(target=self.vumeter, args=())
        self.iteration = 0
        
        self.update_initials()
                               
        dpg.create_context()        
        
        with dpg.font_registry():
            default_font = dpg.add_font("Roboto-Medium.ttf", 16)
            freq_font = dpg.add_font("URWGothic-Book.otf", 32)
        
        with dpg.theme(tag="spec_plot_theme"):
            with dpg.theme_component(0):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 255, 255), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_Fill, (0, 0, 255, 32), category=dpg.mvThemeCat_Plots)
        
        with dpg.theme(tag="vumeter_grid_theme"):
            with dpg.theme_component(0):        
                dpg.add_theme_color(dpg.mvPlotCol_Line, (36, 36, 36, 255), category=dpg.mvThemeCat_Plots, tag="h_line_color")
                
        with dpg.theme(tag="vumeter_bar_theme"):
            with dpg.theme_component(0):        
                dpg.add_theme_color(dpg.mvPlotCol_Fill, (0, 192, 255, 255), category=dpg.mvThemeCat_Plots, tag="bar_color")   
        
        with dpg.theme(tag="tune_freq_color_theme"):
            with dpg.theme_component(0):        
                dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 0, 0, 192), category=dpg.mvThemeCat_Plots)
                # Yeşilimsi: (0, 255, 192, 127)
                # Mavimsi: (0, 192, 255, 127)

        with dpg.window(tag="Primary Window"):
            dpg.bind_font(default_font)
            with dpg.group(tag="main_group", horizontal=True):
                with dpg.group(tag="display_and_modulation"):
                    with dpg.group(tag="spectrum_and_waterfall_subplots"):                                           
                        with dpg.subplots(2, 1, label="", link_columns=True, height=400, width=500, tag="subplots"): 
                            with dpg.plot(height=-1, width=-1, anti_aliased=True, tag="spectrum"): 
                                self.spec_axis_x = dpg.add_plot_axis(dpg.mvXAxis, tag="spec_axis_x")                               
                                dpg.add_plot_axis(dpg.mvYAxis, tag="spec_axis")                           
                                dpg.set_axis_limits("spec_axis", -100, -20)
                                dpg.add_shade_series(self.spec_axis, self.spectrum, parent="spec_axis", tag="specdata_shade", y2=self.wf_bottom)    
                                dpg.bind_item_theme(dpg.last_item(), "spec_plot_theme")       
                                dpg.add_line_series(self.spec_axis, self.spectrum, parent="spec_axis", tag="specdata_line")
                                dpg.fit_axis_data(self.spec_axis_x)
                                dpg.bind_item_theme(dpg.last_item(), "spec_plot_theme") 
                                dpg.add_vline_series([self.freq], parent="spec_axis", tag="tune_indicator")  
                                dpg.bind_item_theme(dpg.last_item(), "tune_freq_color_theme")              
                            with dpg.plot(no_mouse_pos=True, height=100, width=300, tag="waterfall"):                    
                                with dpg.plot_axis(dpg.mvXAxis, no_gridlines=True, no_tick_marks=True, tag="wfaxis") as self._wfaxis:
                                    dpg.bind_colormap(item="waterfall", source=dpg.mvPlotColormap_Jet)
                                    dpg.add_heat_series(self.waterfall, self.wf_len, self.fft_size, scale_min=-80, scale_max=-40, format='', tag="wfdata", bounds_min=[self.f_span_min, 0], bounds_max=[self.f_span_max, self.wf_len])                    
                                dpg.add_plot_axis(dpg.mvYAxis, no_gridlines=True, no_tick_marks=True)  
                                dpg.fit_axis_data(self._wfaxis)                    
                    with dpg.group(horizontal=True, tag="modulation_buttons", width=85):
                        dpg.add_button(label="AM", callback=self.activate_am_demod, tag="am_button")
                        dpg.add_button(label="info", callback=self.info2)
                        dpg.add_button(label="sdr", callback=self.info)
                        dpg.add_button(label="FM", callback=self.activate_fm_demod, tag="fm_button")                      
                        dpg.add_button(label="btn_down", callback=self.btn_down_clicked, arrow=True, direction=dpg.mvDir_Down)
                        dpg.add_button(label="btn_up", callback=self.btn_up_clicked, arrow=True, direction=dpg.mvDir_Up)
                        dpg.add_checkbox(label="RF NR", callback=self.rf_nr_checked, default_value=False, tag="_rfnr_check")                        
                with dpg.group( tag="settings_and_colorbar"):
                    dpg.add_knob_float(label="   BW", indent=20, min_value=1000, max_value=100000, default_value=self.bw, callback=self.bw_knob_changed, tag="bw_knob")           
                    dpg.add_knob_float(label="RF GAIN", indent=20, callback=self.rf_gain_value, default_value=12, min_value=1, max_value=29, tag="rfgain_knob", tracked=True)                    
                    dpg.add_checkbox(label="RF AGC", callback=self.rf_agc_checked, indent=5, tag="_rfagc_check")
                    dpg.add_checkbox(label="SDR AGC", indent=5, callback=self.sdr_agc_checked, default_value=True)                    
                    dpg.add_colormap_scale(min_scale=-120, max_scale=0, height=200, colormap=dpg.mvPlotColormap_Jet, width=70, pos=(516, 208))            
                with dpg.group(tag="noise_reduction_and_volume"):                      
                    dpg.add_knob_float(label="  SEQ", indent=40, min_value=0, max_value=100, default_value=0, callback=self.rf_nr_knob_changed)
                    dpg.add_knob_float(label="AF NR", indent=40, min_value=0, max_value=100, default_value=0, callback=self.af_nr_knob_changed)                    
                    dpg.add_combo(label="RATE", items=[*self.srates.keys()], callback=self.rate_combo_changed, width=70, default_value=f'{self.srate/1000:.0f}k', indent=20)
                    dpg.add_combo(label="WSIZE", items=[*self.window_sizes.keys()], callback=self.package_size_combo_changed, width=70, default_value="FULL" if self.window_size == self.data_size else f'{self.window_size}', indent=20)                                        
                    dpg.add_text(label="", indent=20)
                    dpg.add_text(label="", default_value="   CENTER FREQUENCY", wrap=80, indent=28)                    
                    dpg.add_input_float(label="", width=142, default_value=self.freq/1e6, format="%.04f", on_enter=True, step=0, tag="freq_input_float", callback=self.freq_float_input_changed)
                    dpg.bind_item_font("freq_input_float", freq_font)
                    with dpg.plot(tag="vumeter", height=85, width=142, no_mouse_pos=True, no_box_select=True, no_menus=True):
                        dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis", no_gridlines=True, no_tick_marks=True, no_tick_labels=True)
                        with dpg.plot_axis(dpg.mvYAxis, tag="levels", no_gridlines=True, no_tick_marks=True, no_tick_labels=True):
                            dpg.add_bar_series(np.arange(10), [10, 20, 30, 40, 50, 70, 40, 30, 20, 10, ], weight=0.88, tag="vumeter_bar_series") 
                            dpg.bind_item_theme(dpg.last_item(), "vumeter_bar_theme")
                            dpg.add_hline_series(np.arange(7, 100, 7), label="horizontal", tag="h_lines")
                            dpg.bind_item_theme(dpg.last_item(), "vumeter_grid_theme")                        
                    dpg.add_slider_int(min_value=0, max_value=100, callback=self.volume_knob_changed, default_value=10, width=142, format="VOLUME")
                    dpg.add_checkbox(label="BIAS-T", callback=self.bias_tee_checked, default_value=False, tag="_bias_tee_check")
        dpg.create_viewport(title='MyRadio', width=764, height=444)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Primary Window", True)
        
        # Prosesler ve aynı zamanda Theadler "deamonize" edildiği zaman, main program sonlanınca
        # onlar da terminate oluyormuş. 
        self.th_streamer.deamon = True
        self.th_demodulator.deamon = True           
        self.th_sound.deamon = True
        self.th_processor.deamon = True
        self.th_updater.deamon = True     
        self.th_vumeter.deamon = True
        
        self.th_streamer.start()        
        self.th_demodulator.start()                    
        self.th_sound.start()
        self.th_processor.start()
        self.th_updater.start()       
        self.th_vumeter.start()
              
        dpg.start_dearpygui()        
        dpg.destroy_context()
        
        # Burada prosesleri ve bir threadi kendi kendilerine sonlandıracak "poison-pill" hazırlanıyor.
        # Her ne kadar yukarıda kendiliklerinden sonlanacağını da yazsam öyle olmuyordu, mecburen ekledim.
        self.qpro.put("kill")
        self.qvu_mes.put("kill")        
        self.qpro_mes.put("kill")                
#         self.demod_parameters["status"] = "kill"
#         self.qdemod_mes.put(self.demod_parameters)                
#         self.sdr_parameters["status"] = "kill"
#         self.qmes.put(self.sdr_parameters)                        
        self.qsnd_mes.put(1)
        
        self.th_updater._stop()
                
    def change_axis_limits(self):
        self.f_span_min = self.freq-self.srate/2
        self.f_span_max = self.freq+self.srate/2
        self.f_step = self.srate/self.fft_size
        self.spec_axis = np.arange(self.f_span_min, self.f_span_max, self.f_step)  
    
    def fit_axis(self):
        dpg.fit_axis_data(self.spec_axis_x)
        dpg.fit_axis_data(self._wfaxis)    
    
    def refresh(self):
        self.change_axis_limits()
        self.fit_axis()
        self.update_graph(1)
        
    def activate_am_demod(self, sender, app_data):
        self.demod_type = "AM"
#         self.demod_parameters["demod_type"] = self.demod_type
        self.f_high = 5000
#         self.demod_parameters["f_high"] = self.f_high
        dpg.configure_item("bw_knob", default_value=self.f_high, max_value=10000)
        self.update_initials()
        if self.qdemod_mes.empty(): 
            self.qdemod_mes.put(1)
        self.info(sender)
        
    def activate_fm_demod(self, sender, app_data):
        self.demod_type = "FM"
#         self.demod_parameters["demod_type"] = self.demod_type
        self.f_high = 16000
#         self.demod_parameters["f_high"] = self.f_high
        dpg.configure_item("bw_knob", default_value=self.f_high, max_value=100000)
        self.update_initials()
        if self.qdemod_mes.empty(): 
#            self.qdemod_mes.put(self.demod_parameters)
           self.qdemod_mes.put(1)
        self.info(sender)
        
    def rf_gain_value(self, sender, app_data):         
        self.rf_gain = self.gain_values[str(int(app_data))]   
#         self.sdr_parameters["rf_gain"] = self.rf_gain
        if self.rf_manual_gain_state:
            if self.qmes.empty(): 
#                 self.qmes.put(self.sdr_parameters)
                self.qmes.put(1)
            self.info(sender)
            
    def rf_agc_checked(self, sender, app_data):       
        self.rf_manual_gain_state = not app_data        
#         self.sdr_parameters["rf_manual_gain_state"] = self.rf_manual_gain_state 
        if self.qmes.empty(): 
#             self.qmes.put(self.sdr_parameters)
            self.qmes.put(1)
        self.info(sender)
    
    def sdr_agc_checked(self, sender, app_data):
        self.agc_state = app_data
#         self.sdr_parameters["sdr_agc"] = self.agc_state
        if self.qmes.empty(): 
#             self.qmes.put(self.sdr_parameters)
            self.qmes.put(1)
        self.info(sender)
    
    def bw_knob_changed(self, sender, app_data):
        self.f_high = app_data
#         self.demod_parameters["f_high"] = self.f_high
        self.update_initials()
        if self.qdemod_mes.empty():
            self.qdemod_mes.put(1)
        self.info(sender)
    
    def rf_nr_knob_changed(self, sender, app_data):
        level = app_data
        if level == 0:
            self.slevel = 0
        else:
            self.slevel = level-101
#             self.demod_parameters["slevel"] = self.slevel
            self.update_initials()
        
        if not self.qdemod_mes.full():
            self.qdemod_mes.put(1)
        self.info(sender)
        
    def af_nr_knob_changed(self, sender, app_data):
        level = int(app_data)
        if level == 0:
            self.alevel = 0
        else:
            self.alevel = level+1
            self.nrw_size = self.alevel
#             self.demod_parameters["alevel"] = self.alevel
#             self.demod_parameters["nrw_size"] = self.nrw_size
            self.update_initials()

        if self.qdemod_mes.empty():
            self.qdemod_mes.put(1)
        self.info(sender)
    
    def rate_combo_changed(self, sender, app_data): 
        with self.lock:
            self.srate = self.srates[app_data]
#             self.sdr_parameters["srate"] = self.srate
#             self.demod_parameters["srate"] = self.srate
            self.decimation_factor = int(np.rint(self.srate/self.arate))
#             self.demod_parameters["decimation_factor"] = self.decimation_factor
            self.chunk_size = int(np.rint(self.arate*(self.data_size/self.srate)))
#             self.demod_parameters["chunk_size"] = self.chunk_size
            self.fm_rfnr_size = 12 if self.srate == 960000 else 23
            self.am_rfnr_size = 131 if self.srate == 960000 else 225
            self.update_initials()
                     
        self.refresh()
                
        if self.qmes.empty(): 
#             self.qmes.put(self.sdr_parameters)
            self.qmes.put(1)
        if self.qdemod_mes.empty():
            self.qdemod_mes.put(1)
        if self.qsnd_mes.empty():
            self.qsnd_mes.put(1)
               
        self.th_sound.join()               
        self.event.clear()           
        self.th_sound = th.Thread(target=self.sound, args=())
        self.th_sound.start()
        self.info(sender)
                               
    def package_size_combo_changed(self, sender, app_data):        
        self.window_size = self.data_size if app_data == "FULL" else self.window_sizes[app_data]        
#         self.demod_parameters["window_size"] = self.window_size 
        self.update_initials()
        if self.qdemod_mes.empty():
            self.qdemod_mes.put(1)
        self.info(sender)
    
    def btn_up_clicked(self, sender):
        self.freq = self.freq + 100000 if self.demod_type == "FM" else self.freq + 5000
#         self.sdr_parameters["freq"] = self.freq
        if self.qmes.empty(): 
#             self.qmes.put(self.sdr_parameters)
            self.qmes.put(1)
        dpg.configure_item("freq_input_float", default_value=self.freq/1e6)              
        self.change_axis_limits()
        self.fit_axis()
        self.update_graph(1)
                
    def btn_down_clicked(self, sender):
        self.freq = self.freq - 100000 if self.demod_type == "FM" else self.freq - 5000
#         self.sdr_parameters["freq"] = self.freq
        if self.qmes.empty(): 
#             self.qmes.put(self.sdr_parameters)
            self.qmes.put(1)

        dpg.configure_item("freq_input_float", default_value=self.freq/1e6)
        self.change_axis_limits()
        self.fit_axis()
        self.update_graph(1)
        
    def freq_float_input_changed(self, sender, app_data):
        try:
            self.freq = int(app_data*1e6)
#             self.sdr_parameters["freq"] = self.freq
            if self.qmes.empty(): 
#                 self.qmes.put(self.sdr_parameters)
                self.qmes.put(1)
            self.refresh()
            self.info(sender)
        except:
            print("frequency not updated !")
    
    def volume_knob_changed(self, sender, app_data):
        self.volume = app_data        
        
    def rf_nr_checked(self, sender, app_data):
        self.rf_nr_state = app_data
#         self.demod_parameters["rf_nr"] = self.rf_nr_state
        if self.qdemod_mes.empty():
            self.qdemod_mes.put(1)
        self.info(sender)

    def bias_tee_checked(self, sender, app_data):
        self.bias_tee_state = app_data
        self.sdr.set_bias_tee(app_data)
                
    def info2(self, sender):
        info2 = (f'{"* Demodulator Parameters:"}\n'
                  f'{"* Type":<20} {"None" if self.demod_type == None else self.demod_type:>10} \n'
                  f'{"* f_low":<20} {self.f_low:>10.0f} Hz\n'
                  f'{"* f_high":<20} {1e-3*self.f_high:>10.1f} kHz\n'
                  f'{"* Data size":<20} {self.data_size:>10.0f} Sample\n'                  
                  f'{"* Chunk size":<20} {self.chunk_size:>10.0f} Sample\n'
                  f'{"* Decimating factor":<20} {self.decimation_factor:>10.0f}\n' 
                  f'{"* Conv. window size":<20} {self.window_size:>10.0f}\n'
                  f'{"* AF NR window size":<20} {self.nrw_size:>10.0f}\n'
                  f'{"* RF NR window size":<20} {"None" if self.demod_type == None else self.am_rfnr_size if self.demod_type == "AM" else self.fm_rfnr_size:>10.0f}\n'
                  f'{"* Squelch level":<20} {self.slevel:>10.0f} dB\n'                    
                  )
        print(info2)
                        
    def info(self, sender):
        os.system("clear")
        info = (f'{"   MY DearPyGUI RADIO   ":*^80}\n'
                  f'{"* Frequency":<20} {1e-6*self.freq:>10.04f} MHz\n'
                  f'{"* Sampling rate":<20} {1e-3*self.srate:>10.0f} kSps\n'
                  f'{"* Bandwidth":<20} {1e-3*self.f_high:>10.1f} kHz\n'
                  f'{"* FFT Size":<20} {self.fft_size:>10.0f} Sample\n'
                  f'{"* RTL-SDR AGC":<20} {"ON" if self.agc_state else "OFF":>10}\n'
                  f'{"* Tuner Gain":<20} {"AUTO" if not self.rf_manual_gain_state else self.rf_gain:>10}\n'
                  f'{"* Bias-T":<20} {"Enabled" if self.bias_tee_state else "Disabled":>10}\n'
                  )
        print(info)
    
    def update_initials(self):
        # AM ile ilgili olanlar                        
        self.mask_am = self.create_firwin_mask(self.f_low, self.f_high, self.window_size, self.srate, self.windowed).astype(np.complex128)
        self.am_rfnr_buffer = np.zeros(self.data_size+self.am_rfnr_size-1).astype(np.complex128)
        self.am_rfnr_initials = np.zeros(self.am_rfnr_size-1).astype(np.complex128) 
        self.am_rfnr_window = np.ones(self.am_rfnr_size)/self.am_rfnr_size
        
        self.buffer = np.zeros(self.data_size+self.window_size-1).astype(np.complex128)
        self.abuffer = np.zeros(self.chunk_size+self.nrw_size-1).astype(float)
        self.initials = np.zeros(self.window_size-1).astype(np.complex128)
        self.ainitials = np.zeros(self.nrw_size-1).astype(float)
        self.xi_rf, self.yi_rf = 0.0j, 0.0j # Bunlar rf dc_blocker initial değerleri.
        self.xi_af, self.yi_af = 0.0, 0.0 # Bunlar af dc_blocker initial değerleri.
        self.agc_init = 1.0 # Fast_agc için başlangıç değeri                                
        self.nrw_window = np.ones(self.nrw_size)/self.nrw_size
        
        # FM ile ilgili olanlar
        self.mask1_fm = self.create_firwin_mask(0, 80000, self.window_size, self.srate, self.windowed)
        self.mask2_fm = self.create_firwin_mask(self.f_low, self.f_high, self.window_size, self.srate, self.windowed)                  
        self.buffer1 = np.zeros(self.data_size+self.window_size-1).astype(np.complex128)
        self.buffer2 = np.zeros(self.data_size+self.window_size-1).astype(np.complex128)
        self.initial_demod = 0j
        self.initials1 = np.zeros(self.window_size-1).astype(np.complex128)    
        self.initials2 = np.zeros(self.window_size-1).astype(np.complex128)
        self.fm_rfnr_buffer = np.zeros(self.data_size+self.fm_rfnr_size-1).astype(np.complex128)
        self.fm_rfnr_initials = np.zeros(self.fm_rfnr_size-1).astype(np.complex128)    
        self.fm_rfnr_window = np.ones(self.fm_rfnr_size)/self.fm_rfnr_size
                
    def update_graph(self, sender):                
        dpg.set_value('specdata_line', [self.spec_axis, self.spectrum])
        dpg.set_value('specdata_shade', [self.spec_axis, self.spectrum, self.wf_bottom])
        dpg.configure_item('wfdata', bounds_min=[self.f_span_min, 0], bounds_max=[self.f_span_max, self.wf_len])
        dpg.set_value('wfdata', [self.waterfall])
        dpg.set_value('vumeter_bar_series', [np.arange(10), self.vumeter_data])
        dpg.configure_item('tune_indicator', x=[self.freq])
        
    def updater(self):        
        while True:
            try:
                self.spectrum, self.waterfall, self.vumeter_data = self.qpro.get()
                if len(self.waterfall) == self.wf_len:
                    self.update_graph(1)                    
            except:
                print("updater error")
                
    def Streaming(self):
        # import asyncio
                    
        # async def async_streaming():
        #     async for samples in self.sdr.stream(num_samples_or_bytes=self.data_size, format="samples", loop=None):                              
        #         if not self.qmes.empty():
        #             parameters = self.qmes.get()
        #             freq, rf_manual_gain_state, gain, sdr_agc_mode, srate, status = parameters.values()                              
        #             if status == "kill":
        #                 print("streamer killed")
        #                 break
        #             self.sdr.set_sample_rate(srate)
        #             self.sdr.set_direct_sampling(0 if freq>28800000 else 2)
        #             self.sdr.set_center_freq(freq)                
        #             self.sdr.set_gain("auto" if not rf_manual_gain_state else gain)                
        #             self.sdr.set_agc_mode(sdr_agc_mode)                    
                
        #         if not self.qsdr.full(): self.qsdr.put(samples)   
        #         if not self.qspwf.full(): self.qspwf.put(samples)  
                        
        # loop = asyncio.get_event_loop()
        # loop.run_until_complete(async_streaming())
        # print("streamingde sıkıntı çıktı !")
        while True:
            try:
                if not self.qmes.empty():
                    self.qmes.get()
                    with self.lock:
                        self.sdr.set_sample_rate(self.srate)
                        self.sdr.set_direct_sampling(0 if self.freq>28800000 else 2)
                        self.sdr.set_center_freq(self.freq)                
                        self.sdr.set_gain("auto" if not self.rf_manual_gain_state else self.rf_gain)                
                        self.sdr.set_agc_mode(self.agc_state)         
                samples = self.sdr.read_samples(self.data_size)                
                self.qsdr.put(samples)                        
                if not self.qspwf.full(): self.qspwf.put(samples)
            except:
                print("streamingde sıkıntı")
    
    def squelch(self, data, level):      
        plevel = 10*np.log10(np.var(data))
        print(level, plevel)
        return True if plevel < level else False            
    
    def create_firwin_mask(self, f_low, f_high, window_size, rate, windowed, window="nuttall"):
        if self.f_low == 0: 
            return s.firwin(window_size, f_high, window="nuttall", fs=rate, pass_zero="lowpass")
        else:
            return s.firwin(window_size, [f_low, f_high], window="nuttall", fs=rate, pass_zero="bandpass")
        
    def demodulate_am(self, x):
        # RF DC Bloklama, bu önemli.. IQ Correction da deniyor..
        x, self.xi_rf, self.yi_rf = hpf_ran.hpf(x, self.xi_rf, self.yi_rf, alpha=0.99999)
        
        #  RF Squelch..
        if self.slevel:
           if self.squelch(x, self.slevel):  
#                self.mask_am.fill(0)
#                x.fill(0)
                x = np.zeros(self.data_size).astype(np.complex128)
        
        # AGC parametreleri burada ayarlanıyor. 
        min_level_db = -70.0
        max_level_db = -10.0
        min_gain = 1.0
        max_gain = 10.0
        
        gain = agc_ran.my_agc(x, min_level_db, max_level_db, min_gain, max_gain)
        x *= gain    
        
        # Burada AM RF Noise Reduction yapılıyor, ama denemek gerek, belki de pencere boyu bant genişiliği
        # parametresine de bağlanabilir. 
        if self.rf_nr_state:
            self.am_rfnr_buffer[:(self.am_rfnr_size-1)] = self.am_rfnr_initials
            self.am_rfnr_buffer[-self.data_size:] = x
            x = moving_average_ran.moving_average_fast_cmpx(self.am_rfnr_buffer, self.am_rfnr_window)
            self.am_rfnr_initials = self.am_rfnr_buffer[-(self.am_rfnr_size-1):]        
        
        # RF kanalı filtreleme..
        self.buffer[:(self.window_size-1)] = self.initials    
        self.buffer[-self.data_size:] = x 
        conv = s.oaconvolve(self.buffer, self.mask_am, mode="valid") # Burda rf bandı filtrelenir..
        
        # AM demodülasyonu..
        demod = np.abs(conv[::self.decimation_factor])
        
        # AF Noise reduction..
        if self.alevel:              
            self.abuffer[:(self.nrw_size-1)] = self.ainitials
            self.abuffer[-self.chunk_size:] = demod
            demod = moving_average_ran.moving_average_fast(self.abuffer, self.nrw_window)
            self.ainitials = self.abuffer[-(self.nrw_size-1):]
        self.initials = x[-(self.window_size-1):] 
        
        # Demodülasyondan gelen DC'nin de süzülmesi..
        demod, self.xi_af, self.yi_af = hpf_ran.hpf(demod, self.xi_af, self.yi_af, alpha=0.999)
        
        # Ses kartına giden veri..
        return demod          
    
    def demodulate_fm(self, x):   
        if self.slevel:
           if self.squelch(x, self.slevel):
#                self.mask1_fm.fill(0)
#                x.fill(0)
               x = np.zeros(self.data_size).astype(np.complex128)
        
        # Burada FM RF Noise Reduction yapılıyor, şimdilik buraya bağladım..
        if self.rf_nr_state:       
            self.fm_rfnr_buffer[:(self.fm_rfnr_size-1)] = self.fm_rfnr_initials
            self.fm_rfnr_buffer[-self.data_size:] = x
            x = moving_average_ran.moving_average_fast_cmpx(self.fm_rfnr_buffer, self.fm_rfnr_window)
#             x = s.oaconvolve(self.fm_rfnr_buffer, self.fm_rfnr_window, mode="valid")
            self.fm_rfnr_initials = self.fm_rfnr_buffer[-(self.fm_rfnr_size-1):]        
        
        # RF kanalı filtrelenir...
        self.buffer1[:(self.window_size-1)] = self.initials1
        self.buffer1[-self.data_size:] = x           
        filtered1 = s.oaconvolve(self.buffer1, self.mask1_fm, mode="valid") # Burda rf bandı filtrelenir.. 
        self.initials1 = x[-(self.window_size-1):]
        
        # RF kanalı demodülasyonu
        t1 = np.insert(filtered1, 0, self.initial_demod)
        demod = np.angle(t1[0:-1] * np.conj(t1[1:])) # Burada zaman domeninde demodüle edildi.
        self.initial_demod = filtered1[-1]
        
        # İlk 15kHz bandında bulunan L+R kanallarının toplamı olan bölge süzülür..
        self.buffer2[:(self.window_size-1)] = self.initials2
        self.buffer2[-self.data_size:] = demod
        filtered2 = s.oaconvolve(self.buffer2, self.mask2_fm, mode="valid")        
        self.initials2 = demod[-(self.window_size-1):]   
        
        # Resample edilir, audio data burası..
        resampled = filtered2[::self.decimation_factor]                  
        
        # Ses kartına giden veri..
        return resampled
            
    def demodulator(self):
        while True:
            try:
                if not self.qdemod_mes.empty():
                    self.qdemod_mes.get()
                    """
                    # AM ile ilgili olanlar                        
                    self.mask_am = self.create_firwin_mask(self.f_low, self.f_high, self.window_size, self.srate, self.windowed).astype(np.complex128)
                    self.am_rfnr_buffer = np.zeros(self.data_size+self.am_rfnr_size-1).astype(np.complex128)
                    self.am_rfnr_initials = np.zeros(self.am_rfnr_size-1).astype(np.complex128) 
                    self.am_rfnr_window = np.ones(self.am_rfnr_size)/self.am_rfnr_size
                    
                    self.buffer = np.zeros(self.data_size+self.window_size-1).astype(np.complex128)
                    self.abuffer = np.zeros(self.chunk_size+self.nrw_size-1).astype(float)
                    self.initials = np.zeros(self.window_size-1).astype(np.complex128)
                    self.ainitials = np.zeros(self.nrw_size-1).astype(float)
                    self.xi_rf, self.yi_rf = 0.0j, 0.0j # Bunlar rf dc_blocker initial değerleri.
                    self.xi_af, self.yi_af = 0.0, 0.0 # Bunlar af dc_blocker initial değerleri.
                    self.agc_init = 1.0 # Fast_agc için başlangıç değeri                                
                    self.nrw_window = np.ones(self.nrw_size)/self.nrw_size
                    
                    # FM ile ilgili olanlar
                    self.mask1_fm = self.create_firwin_mask(0, 80000, self.window_size, self.srate, self.windowed)
                    self.mask2_fm = self.create_firwin_mask(self.f_low, self.f_high, self.window_size, self.srate, self.windowed)                  
                    self.buffer1 = np.zeros(self.data_size+self.window_size-1).astype(np.complex128)
                    self.buffer2 = np.zeros(self.data_size+self.window_size-1).astype(np.complex128)
                    self.initial_demod = 0j
                    self.initials1 = np.zeros(self.window_size-1).astype(np.complex128)    
                    self.initials2 = np.zeros(self.window_size-1).astype(np.complex128)
                    # Alttakiler sadece FM demodülasyonunda giriş datasının RF filtrelenmesi için..
                    self.fm_rfnr_buffer = np.zeros(self.data_size+self.fm_rfnr_size-1).astype(np.complex128)
                    self.fm_rfnr_initials = np.zeros(self.fm_rfnr_size-1).astype(np.complex128)    
                    self.fm_rfnr_window = np.ones(self.fm_rfnr_size)/self.fm_rfnr_size
                    """
                    self.update_initials()
                
                x = self.qsdr.get()
                
                if self.demod_type == "AM":
                    # demod, self.initials, self.ainitials, self.xi_rf, self.yi_rf, self.xi_af, self.yi_af, self.agc_init = self.demodulate_am(x)
                    demod = self.demodulate_am(x)
                                  
                    self.qsnd.put(demod)
                    if not self.qvuin.full(): self.qvuin.put(demod)
                    
                if self.demod_type == "FM": 
                    demod = self.demodulate_fm(x)   
                    self.qsnd.put(demod)
                    if not self.qvuin.full(): self.qvuin.put(demod)
            except:
                print("demodulator error")
    
    def processor(self):   
        vu_data = np.ones(10)*7        
        while True:            
            try:
                if not self.qpro_mes.empty():
                    self.qpro_mes.get()                    
                data = self.qspwf.get()
                if not self.qvuout.empty(): vu_data = self.qvuout.get() 
                # spec, wf_ary = self.spectrum_and_waterfall(data) 
                spec = spectrum_partial_fft_ran.spectrum_partial_fft(data, 1, True, self.fft_size)
                self.waterfall_array.appendleft(spec) 
                wf_ary = np.array(self.waterfall_array)                     
                self.qpro.put((spec, wf_ary, vu_data))                                
            except:
                print("processor error")
    
    def vumeter(self):
        while True:
            try:
                if not self.qvu_mes.empty():
                    self.qvu_mes.get()
                x = self.qvuin.get()            
                levels = calculate_vumeter_data_ran.calculate_vumeter_data_lock_in(np.abs(x).astype(np.float64), self.arate, True, 2048, 100, 7)                
                self.qvuout.put(levels)            
            except:            
                print("vumetrede sıkıntı var!")
    
    def sound(self):                   
        def callback(outdata, frames, time, status):
            try:
                if not self.qsnd_mes.empty():
                    self.qsnd_mes.get()
                    self.event.set()              
                data =  self.qsnd.get() if not self.qsnd.empty() else blank
                outdata[:] = data.real.astype(np.float32).reshape(self.chunk_size, 1)*self.volume*0.1                 
            except:
                print("sorun")
        
        blank = np.zeros(self.chunk_size).astype(np.float32)
        stream = sd.OutputStream(device=sd.default.device, blocksize=self.chunk_size, 
                                  samplerate=self.arate,  channels=1, 
                                  callback=callback, finished_callback=self.event.set)    
        with stream:
            self.event.wait()
            
if __name__ == "__main__":    
    mygui = gui()
    
    
        
            
        