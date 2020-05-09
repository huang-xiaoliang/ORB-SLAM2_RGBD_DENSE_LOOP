/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#include "pointcloudmapping.h"
#include <KeyFrame.h>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h> 
#include "Converter.h"
#include "PointCloude.h"
#include "System.h"

#include <iomanip>

int currentloopcount = 0;
//初始化时启动视图view线程展示点云模型
PointCloudMapping::PointCloudMapping(double resolution_,double meank_,double thresh_)
{
    this->resolution = resolution_;
    this->meank = thresh_;
    this->thresh = thresh_;
    statistical_filter.setMeanK(meank);
    statistical_filter.setStddevMulThresh(thresh);
    voxel.setLeafSize( resolution, resolution, resolution);
    globalMap = boost::make_shared< PointCloud >( );
    
    //viewerThread = make_shared<thread>( bind(&PointCloudMapping::viewer, this ) );
}

void PointCloudMapping::shutdown()
{
    {
        unique_lock<mutex> lck(shutDownMutex);
        shutDownFlag = true;
        //keyFrameUpdated.notify_one();
    }
    //viewerThread->join();
}

void PointCloudMapping::insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth,int idk,vector<KeyFrame*> vpKFs)
{
    /*
    cout<<"receive a keyframe, id = "<<idk<<" 第"<<kf->mnId<<"个"<<endl;
    //cout<<"vpKFs数量"<<vpKFs.size()<<endl;
    unique_lock<mutex> lck(keyframeMutex);
    keyframes.push_back( kf );
    currentvpKFs = vpKFs;
    //colorImgs.push_back( color.clone() );
    //depthImgs.push_back( depth.clone() );
    PointCloude pointcloude;
    pointcloude.pcID = idk;
    //pointcloude.T = ORB_SLAM2::Converter::toSE3Quat( kf->GetPose() );
    pointcloude.pcE = generatePointCloud(kf,color,depth);
    pointcloud.push_back(pointcloude);
    keyFrameUpdated.notify_one();
    */
}

void PointCloudMapping::insertKeyFrame2(KeyFrame* kf, cv::Mat& color, cv::Mat& depth,int idk,vector<KeyFrame*> vpKFs)
{
    if(!loopbusy)
    {
	cout<<"receive a keyframe, id = "<<idk<<" 第"<<kf->mnId<<"个建图关键帧"<<endl;
	cout<<"实际存储的第"<<numKeyFrame+1<<"个存储关键镇"<<endl;
	Eigen::Isometry3d T=ORB_SLAM2::Converter::toSE3Quat( kf->GetPose() );
	PointCloud::Ptr cloud=generatePointCloud(kf,color,depth);
	PointCloud::Ptr cloud2(new PointCloud);
	pcl::transformPointCloud( *cloud, *cloud2, T.inverse().matrix());
	
	
	PointCloude2 pointcloude;
	pointcloude.pcID = idk;
	pointcloude.pcE=cloud2;
	
	//如果正在进行回环，则暂停存储
	pointcloud3.push_back(pointcloude);
	
	
	*globalMap += *cloud2;
	//过滤
	PointCloud::Ptr tmp1 ( new PointCloud );
	statistical_filter.setInputCloud(globalMap);
	statistical_filter.filter( *tmp1 );
	//融合
	PointCloud::Ptr tmp(new PointCloud());
	voxel.setInputCloud( tmp1 );
	voxel.filter( *globalMap );
	//pointcloud2.push_back(pointcloude);
	cout<<"当前关键帧点云数量为："<<pointcloude.pcE->points.size()<<endl;
	cout<<"融合后关键帧点云数量为："<<globalMap->points.size()<<endl;
	
	numKeyFrame++;
    }
}

void PointCloudMapping::saveKeyFramePointCloud(const string &filename,PointCloud::Ptr p)
{
    //如果点云集合为空则返回
    if(p->empty())
    {
      return;
    }
    cout << endl << "Saving keyFrame pointclod to " << filename << " ..." << endl;
    pcl::io::savePCDFileBinary(filename, *p );
}

void PointCloudMapping::mergeAllKeyFrame()
{
  PointCloud::Ptr mergeP(new PointCloud);
  for(int i=1;i<=numKeyFrame;i++)
  {
    
    std::ostringstream ostr2;
    ostr2<<i<<"frame.pcd";
    string filename=ostr2.str();
    cout<<"读取文件："<<filename<<endl;
    try
    {
      PointCloud::Ptr p (new PointCloud);
      pcl::io::loadPCDFile(filename,*p);
      *mergeP += *p;
      //过滤
      PointCloud::Ptr tmp1 ( new PointCloud );
      statistical_filter.setInputCloud(mergeP);
      statistical_filter.filter( *tmp1 );
      //融合
      PointCloud::Ptr tmp(new PointCloud());
      voxel.setInputCloud( tmp1 );
      voxel.filter( *mergeP );
    }
    catch(pcl::IOException e)
    {
      cout<<"读取文件异常，结束任务"<<endl;
      break;
    }
    
  }
  try
  {
      pcl::io::savePCDFileBinary("map.pcd", *mergeP );
  }
  catch(pcl::IOException e)
  {
    cout<<"存储文件异常，结束任务"<<endl;
    return;
  }
  
}

pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)//,Eigen::Isometry3d T
{
    PointCloud::Ptr tmp( new PointCloud() );
    // point cloud is null ptr
    for ( int m=0; m<depth.rows; m+=3 )
    {
        for ( int n=0; n<depth.cols; n+=3 )
        {
            float d = depth.ptr<float>(m)[n];
            if (d < 0.01 || d>5)
                continue;
            PointT p;
            p.z = d;
            p.x = ( n - kf->cx) * p.z / kf->fx;
            p.y = ( m - kf->cy) * p.z / kf->fy;
            
            p.b = color.ptr<uchar>(m)[n*3];
            p.g = color.ptr<uchar>(m)[n*3+1];
            p.r = color.ptr<uchar>(m)[n*3+2];
                
            tmp->points.push_back(p);
        }
    }
    
    //Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat( kf->GetPose() );
    //PointCloud::Ptr cloud(new PointCloud);
    //pcl::transformPointCloud( *tmp, *cloud, T.inverse().matrix());
    //cloud->is_dense = false;
    
    //cout<<"generate point cloud for kf "<<kf->mnId<<", size="<<cloud->points.size()<<endl;
    return tmp;
}


void PointCloudMapping::viewer()
{
    pcl::visualization::CloudViewer viewer("viewer");
    while(1)
    {
        
        {
	    
            unique_lock<mutex> lck_shutdown( shutDownMutex );
            if (shutDownFlag)
            {
                break;
            }
        }
        {
	    cout<<"view线程进入睡眠××××××××××××××××××××××××××××××××××××"<<endl;
            unique_lock<mutex> lck_keyframeUpdated( keyFrameUpdateMutex );
            keyFrameUpdated.wait( lck_keyframeUpdated );
        }
        
        // keyframe is updated 
        cout<<"在view线程中显示点云模型×××××××××××××××××××××××××"<<endl;
        viewer.showCloud( globalMap );
	cloudbusy=false;
	/*
	if(numKeyFrame==100)
	{
	    std::ostringstream   ostr;
	    ostr<<numKeyFrame<<"frame.pcd";
	    string filename=ostr.str();
	    pcl::io::savePCDFileBinary(filename, *globalMap );
	    cloudbusy=false;
	}
	*/
	
    }
}
void PointCloudMapping::save()
{
	pcl::io::savePCDFile( "result.pcd", *globalMap );
	cout<<"globalMap save finished"<<endl;
}
//更新点云，在回环loopclosing中调用
void PointCloudMapping::updatecloud()
{
	cout<<"回环重建地图×××××××××××××××××××××××××××××××××××××××××回环重建地图××××××××××××××××××××回环重建"<<endl;
	//如果点云更新不忙，则进行更新，如果忙说明正在运行view线程，则暂停增加点云
	if(!cloudbusy)
	{
	    //将loopbusy设置为true，停止回环
	    loopbusy = true;
	    cout<<"startloopmappoint"<<endl;
	    PointCloud::Ptr tmp1(new PointCloud);
	    //遍历当前关键帧数量，如果点云集合编号和关键帧匹配，则将点云加入tmp1
	    for (int i=0;i<currentvpKFs.size();i++)
	    {
		//遍历关点云集合，
		for (int j=0;j<pointcloud.size();j++)
		{   
		    if(pointcloud[j].pcID==currentvpKFs[i]->mnFrameId) 
		    {   
			  Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat(currentvpKFs[i]->GetPose() );
			  PointCloud::Ptr cloud(new PointCloud);
			  pcl::transformPointCloud( *pointcloud[j].pcE, *cloud, T.inverse().matrix());
			  *tmp1 +=*cloud;
			  //回环过滤
			  pointcloud2.push_back(pointcloud[j]);
			  //cout<<"第pointcloud"<<j<<"与第vpKFs"<<i<<"匹配"<<endl;
			  break;
		    }
		}
	    }
	    
	    //清空pointcloud内容
	    pointcloud.clear();
	    pointcloud.swap(pointcloud2);
	    cout<<"容器pointcloud的容量为："<<pointcloud.size()<<endl;
	    cout<<"容器pointcloud2的容量为："<<pointcloud2.size()<<endl;
	    cout<<"finishloopmap"<<endl;
	    PointCloud::Ptr tmp2(new PointCloud());
	    voxel.setInputCloud( tmp1 );
	    //将过滤后的tmp1存入tmp2
	    voxel.filter( *tmp2 );
	    //融合后的点云
	    globalMap->swap( *tmp2 );
	    //viewer.showCloud( globalMap );
	    loopbusy = false;
	    //cloudbusy = true;
	    //loopcount++表示每次生成点云自增一次
	    loopcount++;

	    //*globalMap = *tmp1;
	}
}
