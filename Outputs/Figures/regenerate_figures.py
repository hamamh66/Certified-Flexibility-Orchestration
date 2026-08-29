import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family':'DejaVu Serif','font.size':15,'axes.titlesize':17,
    'axes.labelsize':16,'xtick.labelsize':14,'ytick.labelsize':14,
    'legend.fontsize':13.5,'axes.linewidth':1.0,'figure.dpi':300,
    'savefig.dpi':400,'axes.grid':True,'grid.alpha':0.28,'grid.linewidth':0.7,
    'axes.axisbelow':True,'legend.frameon':True,'legend.framealpha':0.93,
    'legend.edgecolor':'0.8','mathtext.fontset':'dejavuserif',
})
NAVY='#123A5E'; STEEL='#1B7FB8'; AMBER='#C8871A'; TEAL='#2E8B74'
RED='#B3453A'; GREY='#9AA1A9'; INK='#22262B'

def tidy(ax):
    for sp in ('top','right'): ax.spines[sp].set_visible(False)
    for sp in ('left','bottom'): ax.spines[sp].set_color('0.35')
    ax.tick_params(colors='0.25', length=4)

# ------------------------------------------------------------------ FIG 4
ctl=['B1','B2','B3','B4','B5','B6','P']
sub=['uncontrolled','ToU rule','carbon rule','MPC perfect','MPC realistic','robust MPC','CFO']
delivery=[0,0,0,0.84,0.86,1.35,0.91]
coverage=[0,0,0,0.57,0.40,0.81,0.46]
falseact=[100,100,100,43.2,60.2,19.3,34.1]
netben=[-21.428,-21.428,-21.428,-0.128,0.112,4.392,0.610]
cols=[GREY,GREY,GREY,TEAL,AMBER,AMBER,NAVY]

fig,axes=plt.subplots(1,4,figsize=(21,5.6))
panels=[(delivery,'Flexibility delivery ratio','(a)',1.0),
        (coverage,'Empirical coverage','(b)',0.90),
        (falseact,'False activation (%)','(c)',None),
        (netben,'Net service benefit (k\\$/yr)','(d)',0.0)]
for ax,(vals,title,tag,ref) in zip(axes,panels):
    b=ax.bar(range(7),vals,color=cols,edgecolor='white',linewidth=1.1,width=0.72)
    b[6].set_edgecolor(INK); b[6].set_linewidth(1.8)
    if ref is not None:
        ax.axhline(ref,ls='--',lw=1.3,color='0.45',zorder=1)
    ax.set_title(f'{tag} {title}',pad=12,color=INK)
    ax.set_xticks(range(7))
    ax.set_xticklabels([f'{c}\n{s}' for c,s in zip(ctl,sub)],rotation=45,
                       ha='right',fontsize=11.5)
    ax.grid(axis='x',visible=False)
    for i,v in enumerate(vals):
        off = (max(vals)-min(vals))*0.03
        ax.text(i, v + (off if v>=0 else -off*2.6),
                f'{v:.2f}' if abs(v)<10 else f'{v:.1f}',
                ha='center',va='bottom' if v>=0 else 'top',fontsize=11,color='0.25')
    tidy(ax)
axes[0].set_ylim(0,1.55); axes[1].set_ylim(0,1.0)
axes[2].set_ylim(0,112); axes[3].set_ylim(-25,7)
axes[0].annotate('over-delivery\nvs. depressed baseline',xy=(5,1.35),xytext=(3.1,1.47),
                 fontsize=11,color='0.3',ha='center',
                 arrowprops=dict(arrowstyle='->',color='0.5',lw=1.1))
fig.text(0.5,0.005,'Single realisation; no confidence intervals available '
         '(replication protocol in Sec. 3.9). Rule-based controllers B1–B3 '
         'commit to every request and deliver nothing measurable.',
         ha='center',fontsize=12,color='0.35')
fig.tight_layout(rect=[0,0.045,1,1])
fig.savefig('/home/claude/ms/Figures/fig4_reliability.png',bbox_inches='tight',facecolor='white')
plt.close(fig)

# ------------------------------------------------------------------ FIG 9
r_knob=[0.00,0.05,0.10,0.20,0.30,0.40]
r_del=[0.864,0.992,1.115,1.346,1.576,1.749]
r_ev=[5785,7834,10421,17027,28056,43487]
r_net=[112,1953,2969,4392,5583,6305]
a_knob=[0.30,0.20,0.10,0.05,0.02]
a_del=[0.903,0.901,0.911,0.856,0.835]
a_ev=[5374,4851,4399,4551,4602]
a_net=[992,904,610,-32,-326]

fig,axes=plt.subplots(1,3,figsize=(19,6.0))
ax=axes[0]
ax.plot(np.array(r_ev)/1000,r_del,'-v',color=AMBER,ms=11,lw=2.4,label='robust MPC (knob $r$)')
ax.plot(np.array(a_ev)/1000,a_del,'-o',color=NAVY,ms=10,lw=2.4,label=r'CFO (knob $\alpha$)')
for x,y,k in zip(r_ev,r_del,r_knob):
    ax.annotate(f'$r$={k:.2f}',(x/1000,y),textcoords='offset points',
                xytext=(9,-13),fontsize=11,color=AMBER)
ax.axhline(1.0,ls='--',lw=1.2,color='0.5')
ax.axvspan(3.8,6.2,color=NAVY,alpha=0.07,zorder=0)
ax.text(5.0,1.66,'zoom in (b)',fontsize=11.5,color=NAVY,ha='center')
ax.set_xlabel('unmet EV departure energy (MWh/yr)')
ax.set_ylabel('flexibility delivery ratio')
ax.set_title('(a) The full reliability\u2013mobility frontier',pad=12,color=INK)
ax.legend(loc='lower right'); tidy(ax)
ax.annotate('', xy=(43.5,1.80), xytext=(6.0,1.80),
            arrowprops=dict(arrowstyle='->',color=RED,lw=1.6))
ax.text(24,1.83,r'$\times$7.5 mobility cost buys $\times$2.0 delivery',
        fontsize=12,color=RED,ha='center')
ax.set_ylim(0.78,1.92)

ax=axes[1]
ax.plot(np.array(r_ev)/1000,r_del,'-v',color=AMBER,ms=13,lw=2.6,label='robust MPC')
ax.plot(np.array(a_ev)/1000,a_del,'-o',color=NAVY,ms=12,lw=2.6,label='CFO')
ax.annotate('$r$=0.05',(7.834,0.992),textcoords='offset points',xytext=(-52,4),fontsize=12,color=AMBER)
for x,y,k,off in zip(a_ev,a_del,a_knob,[(0,14),(-4,-22),(-32,2),(-34,-4),(0,-22)]):
    ax.annotate(rf'$\alpha$={k}',(x/1000,y),textcoords='offset points',xytext=off,
                fontsize=12,color=NAVY,ha='center')
ax.set_xlim(4.0,8.7); ax.set_ylim(0.785,1.05)
ax.set_xlabel('unmet EV departure energy (MWh/yr)')
ax.set_ylabel('flexibility delivery ratio')
ax.set_title('(b) Where driver commitments are respected',pad=12,color=INK)
ax.annotate('CFO delivers more, at lower\nmobility cost, than robust MPC\nat its least conservative setting',
            xy=(5.9,0.858),xytext=(7.1,0.822),fontsize=11.5,color=NAVY,ha='center',
            arrowprops=dict(arrowstyle='->',color=NAVY,lw=1.3))
ax.annotate('$r$=0.00',(5.785,0.864),textcoords='offset points',xytext=(-6,16),
            fontsize=12,color=AMBER,ha='center')
ax.legend(loc='upper left'); tidy(ax)

ax=axes[2]
ax.plot(np.array(r_ev)/1000,np.array(r_net)/1000,'-v',color=AMBER,ms=11,lw=2.4,label='robust MPC')
ax.plot(np.array(a_ev)/1000,np.array(a_net)/1000,'-o',color=NAVY,ms=10,lw=2.4,label='CFO')
ax.axhline(0,color='0.4',lw=1.1)
ax.set_xlabel('unmet EV departure energy (MWh/yr)')
ax.set_ylabel('net service benefit (k\\$/yr)')
ax.set_title('(c) Value purchased with driver mobility',pad=12,color=INK)
ax.legend(loc='lower right'); tidy(ax)
fig.text(0.5,0.005,'Upper left is preferred. Each marker is one setting of the '
         "controller family's conservatism knob; Case A, single realisation.",
         ha='center',fontsize=12,color='0.35')
fig.tight_layout(rect=[0,0.05,1,1])
fig.savefig('/home/claude/ms/Figures/fig9_frontier.png',bbox_inches='tight',facecolor='white')
plt.close(fig)

# ------------------------------------------------------------------ FIG 5
nom=[0.70,0.80,0.90,0.95,0.98]
cov_all=[0.467,0.492,0.464,0.396,0.386]
cov_post=[0.567,0.621,0.577,0.444,0.429]
deliv=[4072,3905,3363,2490,2047]
fig,axes=plt.subplots(1,2,figsize=(16,6.2))
ax=axes[0]
ax.plot([0.65,1.0],[0.65,1.0],'--',color='0.55',lw=1.6,label='perfect calibration')
ax.plot(nom,cov_all,'-o',color=NAVY,ms=11,lw=2.6,label='CFO, all events')
ax.plot(nom,cov_post,'-s',color=STEEL,ms=10,lw=2.6,label='CFO, post-commissioning')
for lbl,val,c in [('B6 robust MPC',0.807,AMBER),('B4 MPC perfect',0.567,TEAL),('B5 MPC realistic',0.398,RED)]:
    ax.axhline(val,ls=':',lw=1.6,color=c)
    ax.text(0.655,val+0.012,lbl,fontsize=12,color=c)
ax.annotate('coverage does not rise\nwith nominal level',
            xy=(0.98,0.386),xytext=(0.86,0.30),fontsize=13,color=RED,ha='center',
            arrowprops=dict(arrowstyle='->',color=RED,lw=1.4))
ax.set_xlabel(r'nominal coverage  $1-\alpha$')
ax.set_ylabel('empirical coverage')
ax.set_title('(a) Calibration of the certified envelope',pad=12,color=INK)
ax.set_xlim(0.65,1.005); ax.set_ylim(0.25,1.03)
ax.legend(loc='upper left',bbox_to_anchor=(0.02,0.99)); tidy(ax)

ax=axes[1]
ax.plot(deliv,cov_all,'-o',color=NAVY,ms=11,lw=2.6,label=r'CFO frontier ($\alpha$ sweep)')
for x,y,k,off in zip(deliv,cov_all,[0.30,0.20,0.10,0.05,0.02],
                     [(30,-14),(0,15),(0,15),(-14,15),(-16,-18)]):
    ax.annotate(rf'$\alpha$={k}',(x,y),textcoords='offset points',xytext=off,
                fontsize=12,color=NAVY,ha='center')
for x,y,lbl,c,m in [(4600,0.567,'B4 MPC perfect',TEAL,'D'),
                    (4750,0.398,'B5 MPC realistic',RED,'^'),
                    (7400,0.807,'B6 robust MPC',AMBER,'v')]:
    ax.plot(x,y,m,color=c,ms=15,label=lbl)
ax.annotate('rising conservatism withdraws flexibility\nwithout buying calibration',
            xy=(2100,0.392),xytext=(4300,0.30),fontsize=12.5,color='0.3',ha='center',
            arrowprops=dict(arrowstyle='->',color='0.5',lw=1.3))
ax.set_xlabel('total certified flexibility delivered (kW-events)')
ax.set_ylabel('empirical coverage')
ax.set_title('(b) The coverage\u2013value frontier',pad=12,color=INK)
ax.set_ylim(0.25,0.90); ax.set_xlim(1500,8200)
ax.legend(loc='upper left'); tidy(ax)
fig.tight_layout()
fig.savefig('/home/claude/ms/Figures/fig5_coverage_value.png',bbox_inches='tight',facecolor='white')
plt.close(fig)

# ------------------------------------------------------------------ FIG 7
labs=['Full CFO','$-$ certification','$-$ risk term','$-$ information term','$-$ regime bins']
d=[0.911,0.896,0.911,0.931,0.851]
cv=[0.464,0.432,0.464,0.438,0.404]
fa=[34.1,52.3,34.1,30.7,31.8]
nb=[610,873,610,880,-112]
co2=[283.2,283.6,283.2,283.1,282.7]
cols7=[NAVY,GREY,GREY,GREY,GREY]
fig,axes=plt.subplots(1,4,figsize=(20,5.6))
for ax,(vals,title,tag,fmt) in zip(axes,[
        (d,'Delivery ratio','(a)','{:.3f}'),
        (cv,'Empirical coverage','(b)','{:.3f}'),
        (fa,'False activation (%)','(c)','{:.1f}'),
        (nb,'Net service benefit (\\$/yr)','(d)','{:.0f}')]):
    y=np.arange(5)[::-1]
    bars=ax.barh(y,vals,color=cols7,edgecolor='white',height=0.68)
    bars[0].set_edgecolor(INK); bars[0].set_linewidth(1.6)
    ax.set_yticks(y); ax.set_yticklabels(labs,fontsize=13)
    ax.set_title(f'{tag} {title}',pad=12,color=INK)
    ax.grid(axis='y',visible=False)
    span=max(vals)-min(min(vals),0)
    for yy,v in zip(y,vals):
        ax.text(v+span*0.02 if v>=0 else v-span*0.02, yy, fmt.format(v),
                va='center', ha='left' if v>=0 else 'right', fontsize=12, color='0.25')
    if min(vals)<0: ax.axvline(0,color='0.4',lw=1.0)
    ax.set_xlim(min(min(vals)*1.35,0), max(vals)*1.22)
    tidy(ax)
for ax in axes[1:]: ax.set_yticklabels([])
fig.text(0.5,0.005,'Operational CO$_2$ is invariant across all five variants '
         '(282.7–283.6 t/yr) and is therefore not plotted. Removing the risk term '
         'changes nothing: it never binds. Removing the regime bins is the most damaging ablation.',
         ha='center',fontsize=12,color='0.35')
fig.tight_layout(rect=[0,0.055,1,1])
fig.savefig('/home/claude/ms/Figures/fig7_ablation.png',bbox_inches='tight',facecolor='white')
plt.close(fig)
print('figures written')
