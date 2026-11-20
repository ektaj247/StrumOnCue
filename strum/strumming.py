"""Geometry-aware symbolic notes measured from the simulated right fingertip."""
import numpy as np
import mujoco
from .ukulele import STRING_NAMES, CHORDS, OPEN_MIDI, string_y, string_z, classify_contact, swept_crossings


def score(expected, actual, tolerance=.05):
    matches=[]
    groups={(e['string'],e['pitch'],e['direction']) for e in expected+actual}
    for group in groups:
        left=sorted(e['onset_s'] for e in expected if (e['string'],e['pitch'],e['direction'])==group)
        right=sorted(e['onset_s'] for e in actual if (e['string'],e['pitch'],e['direction'])==group)
        i=j=0
        while i<len(left) and j<len(right):
            delta=right[j]-left[i]
            if delta < -tolerance: j+=1
            elif delta > tolerance: i+=1
            else: matches.append(delta);i+=1;j+=1
    tp=len(matches)
    return {'tp':tp,'fp':len(actual)-tp,'fn':len(expected)-tp,
            'f1':2*tp/(len(expected)+len(actual)) if expected or actual else 0.,
            'matched_mean_absolute_onset_error_ms':float(np.mean(np.abs(matches))*1000) if matches else None}


class Strummer:
    def __init__(self,m,d,p,schedule):
        self.m,self.p=m,p
        self.uke=m.body('ukulele').id
        self.origin=d.xpos[self.uke].copy()
        self.rotation=d.xmat[self.uke].reshape(3,3).copy()
        self.x=.29*p.scale_length/.38
        self.y0=string_y(p,0,self.x)-.009
        self.y1=string_y(p,3,self.x)+.009
        self.z=string_z(p,self.x)+.0007
        self.safe_z=.06
        self.act=[m.actuator('right_stage_'+x+'_servo').id for x in 'xyz']
        self.q=[m.joint('right_stage_'+x).qposadr[0] for x in 'xyz']
        self.previous=self.local(d.site('right_strum_tip').xpos)
        self.last={s:-1. for s in STRING_NAMES}
        self.strokes=[];self.expected=[];self.actual=[]
        start=0.
        for cycle,(name,seconds,src,dst) in schedule:
            if name.startswith('hold_') or name=='returned_C':
                for beat in range(3):
                    onset=start+1.+beat*60/70
                    self.strokes.append(onset)
                    for i,s in enumerate(STRING_NAMES):
                        self.expected.append({'cycle':cycle,'chord':dst,'string':s,'pitch':OPEN_MIDI[i]+CHORDS[dst][i],
                          'direction':'D','onset_s':onset+.3*(string_y(p,i,self.x)-self.y0)/(self.y1-self.y0)})
            start+=seconds
        self.string_ids={m.geom('string_'+s).id:s for s in STRING_NAMES}
    def local(self,point):
        return self.rotation.T@(point-self.origin)
    def desired(self,t):
        def smooth(u):
            u=np.clip(u,0,1);return u*u*(3-2*u)
        for start in self.strokes:
            dt=t-start
            if -.25<=dt<0: local=[self.x,self.y0,self.safe_z+(self.z-self.safe_z)*smooth((dt+.25)/.25)]
            elif 0<=dt<=.3: local=[self.x,self.y0+(self.y1-self.y0)*dt/.3,self.z]
            elif .3<dt<=.45: local=[self.x,self.y1,self.z+(self.safe_z-self.z)*smooth((dt-.3)/.15)]
            elif .45<dt<=.55: local=[self.x,self.y1+(self.y0-self.y1)*smooth((dt-.45)/.1),self.safe_z]
            else: continue
            return self.origin+self.rotation@np.array(local)
        return self.origin+self.rotation@np.array([self.x,self.y0,self.safe_z])
    def control(self,d):
        t=float(d.time)+self.m.opt.timestep
        velocity=(self.desired(t+.0001)-self.desired(t-.0001))/.0002
        d.ctrl[self.act]=d.qpos[self.q]+self.desired(t)-d.site('right_strum_tip').xpos+.025*velocity
    def observe(self,d):
        m,p=self.m,self.p
        active={s:{} for s in STRING_NAMES};wrench=np.zeros(6)
        for k,c in enumerate(d.contact):
            pair={int(c.geom1),int(c.geom2)}
            sid=next((i for i in pair if i in self.string_ids),None)
            if sid is None:continue
            other=next((i for i in pair if i!=sid),None)
            if other is None or not m.body(int(m.geom_bodyid[other])).name.startswith('lh_'):continue
            label=classify_contact(p,self.local(c.pos))
            if label is None or label[0]!=self.string_ids[sid]:continue
            mujoco.mj_contactForce(m,d,k,wrench)
            s,n=label;active[s][n]=active[s].get(n,0.)+max(0.,float(wrench[0]))
        frets={s:max([n for n,f in active[s].items() if f>=.01],default=0) for s in STRING_NAMES}
        tip=self.local(d.site('right_strum_tip').xpos)
        t0=float(d.time)-m.opt.timestep
        if abs(tip[1]-self.previous[1])/m.opt.timestep>=.05:
            for event in swept_crossings(p,self.previous,tip):
                s=event['string'];onset=t0+event['fraction']*m.opt.timestep
                if onset-self.last[s]<.03:continue
                self.last[s]=onset
                self.actual.append({'string':s,'pitch':OPEN_MIDI[STRING_NAMES.index(s)]+frets[s],
                  'direction':event['direction'],'onset_s':onset,'detected_fret':frets[s]})
        self.previous=tip
    def summary(self):
        return {'notes':score(self.expected,self.actual),'expected_events':len(self.expected),'actual_events':len(self.actual),
          'unexpected_upstrokes':sum(e['direction']=='U' for e in self.actual),
          'bpm_within_hold':70,'strums_per_hold':3,'readiness_gating':False,
          'scope':'Stage-actuated right hand. Right/string collisions disabled. Events use actual fingertip crossings and contact-derived pitches; no string acoustics.',
          'right_hand_settings':{'stage_axes':3,'gravity_compensation':True,'integrator':'implicitfast','stage_kp':8000,'stage_kv':200,'stage_force_limit_N':200,'velocity_feedforward_s':.025},
          'height_tolerance_m':.002,'force_threshold_N':.01,'debounce_s':.03,'matching_tolerance_s':.05}
